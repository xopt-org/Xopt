import logging
from typing import Callable, Dict, List

import pandas as pd
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    InverseCostWeightedUtility,
    PosteriorMean,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import AffineFidelityCostModel, SingleTaskMultiFidelityGP
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch import Module
from pydantic import BaseModel, create_model, Field, root_validator

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.models.multi_fidelity import create_multifidelity_model
from xopt.generators.bayesian.objectives import create_mc_objective
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions, ModelOptions
from xopt.pydantic import JSON_ENCODERS
from xopt.utils import format_option_descriptions, get_function, get_function_defaults
from xopt.vocs import VOCS

logger = logging.getLogger()


class MultiFidelityModelOptions(ModelOptions):
    """Options for defining the Multi-fidelity GP model in BO"""

    function: Callable
    fidelity_key: str = Field("s", description="fieldity parameter name")
    kwargs: BaseModel

    class Config:
        arbitrary_types_allowed = True
        json_encoders = JSON_ENCODERS
        extra = "forbid"
        allow_mutation = True

    @root_validator(pre=True)
    def validate_all(cls, values):
        if "function" in values.keys():
            f = get_function(values["function"])
        else:
            f = create_multifidelity_model

        kwargs = values.get("kwargs", {})
        kwargs = {**get_function_defaults(f), **kwargs}
        values["function"] = f
        values["kwargs"] = create_model("kwargs", **kwargs)()

        return values


class MultiFidelityAcqOptions(AcqOptions):
    fidelities: list = Field(
        [0.5, 0.75, 1.0],
        description="fixed fidelities to evaluate the acquisiton function on (less "
        "values equals better computational perfromance",
    )
    n_fantasies: int = Field(
        128,
        description="number of fantasy samples to take for "
        "calculating knowledge gradient",
        ge=2,
    )
    base_cost: float = Field(
        1.0, description="base cost added to fidelity cost of running a sample"
    )


class MultiFidelityOptions(BayesianOptions):
    model = MultiFidelityModelOptions()
    acq = MultiFidelityAcqOptions()


class MultiFidelityBayesianGenerator(BayesianGenerator):
    alias = "multi_fidelity"
    __doc__ = (
        """Implements Multi-fidelity Bayeisan optimizationn
        
        Assumes a fidelity parameter [0,1]
        
        """
        + f"{format_option_descriptions(MultiFidelityOptions())}"
    )

    def __init__(self, vocs: VOCS, options: MultiFidelityOptions = None):
        """
        Generator using Expected improvement acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: BayesianOptions
            Specific options for this generator
        """
        options = options or MultiFidelityOptions()
        if not type(options) is MultiFidelityOptions:
            raise ValueError("options must be a `MultiFidelityOptions` object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)

    @staticmethod
    def default_options() -> MultiFidelityOptions:
        return MultiFidelityOptions()

    def generate(self, n_candidates: int) -> List[Dict]:
        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            tkwargs = {"dtype": torch.double, "device": "cpu"}

            # update internal model with internal data
            self.train_model(self.data)

            acq_funct = self.get_acquisition(self._model)

            # get candidates in real domain at discrete fidelities specified by options
            ff_list = [
                {self.fidelity_index: ele} for ele in self.options.acq.fidelities
            ]
            candidates, out = optimize_acqf_mixed(
                acq_function=acq_funct,
                bounds=self._get_bounds(),
                fixed_features_list=ff_list,
                q=n_candidates,
                num_restarts=self.options.optim.num_restarts,
                raw_samples=self.options.optim.raw_samples,
                # batch_initial_conditions=X_init,
                options={"batch_limit": 5, "maxiter": 200},
            )
            logger.debug("Best candidate from optimize", candidates, out)

            # build candidate data frame with fidelity parameter
            variable_values = self.vocs.convert_numpy_to_inputs(
                candidates[:, :-1].detach().numpy()
            )
            fidelity_values = pd.DataFrame(
                candidates[:, -1:].detach().numpy(), columns=[self.fidelity_key]
            )

            return pd.concat((variable_values, fidelity_values), axis=1)

    def calculate_total_cost(self, data: pd.DataFrame = None) -> float:
        # calculate total cost of data samples using the fidelity parameter
        if data is None:
            data = self.data

        f_data = data[self.fidelity_key].to_numpy()
        return f_data.sum() + self.options.acq.base_cost * len(f_data)

    def _get_acquisition(self, model):
        """
        Creates the Multi-Fidelity Knowledge Gradient acquistion function

        In order for MFKG to evaluate the information gain, it uses the model to
        predict the function value at the highest fidelity after conditioning
        on the observation. This is handled by the project argument, which specifies
        how to transform a tensor X to its target fidelity. We use a default helper
        function called project_to_target_fidelity to achieve this.

        An important point to keep in mind: in the case of standard KG, one can ignore
        the current value and simply optimize the expected maximum posterior mean of the
        next stage. However, for MFKG, since the goal is optimize information gain per
        cost, it is important to first compute the current value (i.e., maximum of the
        posterior mean at the target fidelity). To accomplish this, we use a
        FixedFeatureAcquisitionFunction on top of a PosteriorMean.

        """
        cost_model = AffineFidelityCostModel(
            fidelity_weights=self.target_fidelity, fixed_cost=5.0
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=self.vocs.n_variables + 1,
            columns=[self.fidelity_index],
            values=[1],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self._get_bounds()[:, :-1],
            q=1,
            num_restarts=self.options.optim.num_restarts,
            raw_samples=self.options.optim.raw_samples,
            options={"batch_limit": 10, "maxiter": 200},
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.options.acq.n_fantasies,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=self._project,
        )

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.data

        # drop nans
        valid_data = data[
            pd.unique(
                self.vocs.variable_names + self.vocs.output_names + [self.fidelity_key]
            )
        ].dropna()

        kwargs = self.options.model.kwargs.dict()

        _model = self.options.model.function(
            valid_data, self.vocs, self.fidelity_key, **kwargs
        )

        # validate returned model
        self._validate_model(_model)

        if update_internal:
            self._model = _model
        return _model

    def _get_objective(self):
        return create_mc_objective(self.vocs)

    @property
    def fidelity_key(self):
        return self.options.model.fidelity_key

    @property
    def fidelity_index(self):
        return self.vocs.n_variables

    @property
    def target_fidelity(self):
        return {self.fidelity_index: 1.0}

    def get_acquisition(self, model):
        acq = self._get_acquisition(model)
        return acq

    def _project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelity)

    def _get_bounds(self) -> torch.Tensor:
        bounds = super()._get_bounds()
        mf_bounds = torch.hstack((bounds, torch.tensor([0, 1]).reshape(2, 1)))
        return mf_bounds

    def _validate_model(self, model):
        if not isinstance(model, SingleTaskMultiFidelityGP):
            raise ValueError("model must be SingleTaskMultiFidelityGP object")
