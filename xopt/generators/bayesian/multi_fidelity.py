import logging
from copy import deepcopy
from typing import Callable, List

import pandas as pd
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    GenericMCObjective,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from pydantic import Field

from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.multi_fidelity import NMOMF

from xopt.generators.bayesian.mobo import MOBOGenerator, MOBOOptions
from xopt.generators.bayesian.options import AcqOptions, ModelOptions
from xopt.utils import format_option_descriptions
from xopt.vocs import ObjectiveEnum, VOCS

logger = logging.getLogger()


class MultiFidelityModelOptions(ModelOptions):
    """Options for defining the Multi-fidelity GP model in BO"""

    name: str = Field("standard", description="name of model constructor")
    fidelity_parameter: str = Field("s", description="fidelity parameter name")


class MultiFidelityAcqOptions(AcqOptions):
    cost_function: Callable = Field(
        lambda x: x + 1.0,
        description="callable function that describes the cost "
        "of evaluating the objective function",
    )
    reference_point: List[float] = Field(
        None,
        description="reference point for multi-objective, multi-fidelity "
        "optimization",
    )


class MultiFidelityOptions(MOBOOptions):
    acq = MultiFidelityAcqOptions()
    model = MultiFidelityModelOptions()


class MultiFidelityGenerator(MOBOGenerator):
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

        # create an augmented vocs that includes the fidelity parameter for both the
        # variable and constraint
        _vocs = deepcopy(vocs)
        _vocs.variables[options.model.fidelity_parameter] = [0, 1]
        _vocs.objectives[options.model.fidelity_parameter] = ObjectiveEnum("MAXIMIZE")

        super().__init__(_vocs, options)

    @staticmethod
    def default_options() -> MultiFidelityOptions:
        return MultiFidelityOptions()

    def calculate_total_cost(self, data: pd.DataFrame = None) -> float:
        """calculate total cost of data samples using the fidelity parameter"""
        if data is None:
            data = self.data

        f_data = self.get_input_data(data)

        # apply callable function to get costs
        return self.options.acq.cost_function(
            f_data[..., self.fidelity_variable_index]
        ).sum()

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

        X_baseline = self.get_input_data(self.data)

        # wrap the cost function such that it only has to accept the fidelity parameter
        def true_cost_function(X):
            return self.options.acq.cost_function(X[..., self.fidelity_variable_index])

        acq_func = NMOMF(
            model=model,
            X_baseline=X_baseline,
            ref_point=self.reference_point,
            cost_call=true_cost_function,
            objective=self._get_objective(),
            constraints=self._get_constraint_callables(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq_func

    def add_data(self, new_data: pd.DataFrame):
        # overwrite add data to check for valid fidelity values
        if (new_data[self.fidelity_parameter] > 1.0).any() or (
            new_data[self.fidelity_parameter] < 0.0
        ).any():
            raise ValueError("cannot add fidelity data that is outside the range [0,1]")
        super().add_data(new_data)

    @property
    def fidelity_parameter(self):
        return self.options.model.fidelity_parameter

    @property
    def fidelity_variable_index(self):
        return self.vocs.variable_names.index(self.fidelity_parameter)

    @property
    def fidelity_objective_index(self):
        return self.vocs.objective_names.index(self.fidelity_parameter)

    def get_optimum(self):
        """select the best point at the maximum fidelity"""

        # define single objective based on vocs
        weights = torch.zeros(self.vocs.n_outputs, **self._tkwargs)
        for idx, ele in enumerate(self.vocs.objective_names):
            if self.vocs.objectives[ele] == "MINIMIZE":
                weights[idx] = -1.0
            elif self.vocs.objectives[ele] == "MAXIMIZE":
                weights[idx] = 1.0

        def obj_callable(Z):
            return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=GenericMCObjective(obj_callable)
            ),
            self._get_constraint_callables(),
        )

        max_fidelity_c_posterior_mean = FixedFeatureAcquisitionFunction(
            c_posterior_mean,
            self.vocs.n_variables,
            [self.fidelity_variable_index],
            [1.0],
        )

        boundst = self._get_bounds().T
        fixed_bounds = torch.cat(
            (
                boundst[: self.fidelity_variable_index],
                boundst[self.fidelity_variable_index + 1:],
            )
        ).T

        result, out = optimize_acqf(
            acq_function=max_fidelity_c_posterior_mean,
            bounds=fixed_bounds,
            q=1,
            raw_samples=self.options.optim.raw_samples * 5,
            num_restarts=self.options.optim.num_restarts * 5,
        )
        vnames = deepcopy(self.vocs.variable_names)
        del vnames[self.fidelity_variable_index]
        df = pd.DataFrame(result.detach().cpu().numpy(), columns=vnames)
        df[self.fidelity_parameter] = 1.0

        return self.vocs.convert_dataframe_to_inputs(df)

    @property
    def reference_point(self):
        # case for multi-fidelity multi-objective
        pt = []
        for name in self.vocs.objective_names:
            if name == self.fidelity_parameter:
                ref_val = 0.0
            else:
                # if this is a single objective problem then there will be no
                # reference point in the acq options
                if self.options.acq.reference_point is None:
                    ref_val = 10.0
                else:
                    ref_val = self.options.acq.reference_point[name]

            if self.vocs.objectives[name] == "MINIMIZE":
                pt += [-ref_val]
            elif self.vocs.objectives[name] == "MAXIMIZE":
                pt += [ref_val]
            else:
                raise ValueError(
                    f"objective type {self.vocs.objectives[name]} not\
                        supported"
                )

        return torch.tensor(pt, **self._tkwargs)
