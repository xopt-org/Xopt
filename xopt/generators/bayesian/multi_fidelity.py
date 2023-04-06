import logging
from copy import deepcopy
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, qUpperConfidenceBound, \
    GenericMCObjective
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.optim import optimize_acqf
from pydantic import Field

from xopt.errors import XoptError
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.custom_botorch.constrained_acqusition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.multi_fidelity import NMOMF
from xopt.generators.bayesian.objectives import create_momf_objective
from xopt.generators.bayesian.options import AcqOptions, BayesianOptions, ModelOptions
from xopt.generators.bayesian.utils import compute_hypervolume
from xopt.utils import format_option_descriptions
from xopt.vocs import VOCS

logger = logging.getLogger()


class MultiFidelityModelOptions(ModelOptions):
    """Options for defining the Multi-fidelity GP model in BO"""

    name: str = Field("multi_fidelity", description="name of model constructor")
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


class MultiFidelityOptions(BayesianOptions):
    acq = MultiFidelityAcqOptions()
    model = MultiFidelityModelOptions()


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

        super().__init__(vocs, options, supports_batch_generation=True)

        # create an augmented vocs that includes the fidelity parameter for both the
        # variable and constraint
        self._vocs = deepcopy(self.vocs)
        self._vocs.variables[self.fidelity_parameter] = [0,1]
        self._vocs.objectives[self.fidelity_parameter] = "MAXIMIZE"

    @staticmethod
    def default_options() -> MultiFidelityOptions:
        return MultiFidelityOptions()

    def calculate_total_cost(self, data: pd.DataFrame = None) -> float:
        """calculate total cost of data samples using the fidelity parameter"""
        if data is None:
            data = self.data

        f_data = self.get_input_data(data)

        # apply callable function to get costs
        return self.options.acq.cost_function(f_data[..., -1]).sum()

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
            return self.options.acq.cost_function(X[..., -1])

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

    def _get_objective(self):
        return create_momf_objective(self.vocs, self._tkwargs)

    def _get_optimization_bounds(self):
        """
        gets optimization bounds including fidelity parameter

        """
        bounds = self._get_bounds()
        mf_bounds = torch.hstack((bounds, torch.tensor([0, 1]).reshape(2, 1)))
        return mf_bounds

    def add_data(self, new_data: pd.DataFrame):
        # overwrite add data to check for valid fidelity values
        if (new_data[self.fidelity_parameter] > 1.0).any() or (
            new_data[self.fidelity_parameter] < 0.0
        ).any():
            raise ValueError("cannot add fidelity data that is outside the range [0,1]")
        self.data = pd.concat([self.data, new_data], axis=0)

    def get_input_data(self, data):
        return torch.tensor(
            data[self.vocs.variable_names + [self.fidelity_parameter]].to_numpy(),
            **self._tkwargs,
        )

    def _process_candidates(self, candidates):
        logger.debug("Best candidate from optimize", candidates)
        result = self.vocs.convert_numpy_to_inputs(
            candidates[..., :-1].detach().cpu().numpy()
        )
        # add fidelity parameter
        result[self.fidelity_parameter] = candidates[..., -1].detach().cpu().numpy()

        return result

    @property
    def fidelity_parameter(self):
        return self.options.model.fidelity_parameter

    @property
    def fidelity_index(self):
        return self.vocs.n_variables

    def _validate_model(self, model):
        if not isinstance(model, SingleTaskGP):
            raise ValueError("model must be SingleTaskGP object")

    def get_optimum(self):
        """select the best point at the maximum fidelity"""

        # define single objective based on vocs
        weights = torch.zeros(self.vocs.n_outputs + 1, **self._tkwargs)
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
            c_posterior_mean, self.vocs.n_variables + 1, [self.vocs.n_variables], [1.0]
        )

        result, out = optimize_acqf(
            acq_function=max_fidelity_c_posterior_mean,
            bounds=self._get_bounds(),
            q=1,
            raw_samples=self.options.optim.raw_samples * 5,
            num_restarts=self.options.optim.num_restarts * 5,
        )

        return self.vocs.convert_numpy_to_inputs(
            result.detach().cpu().numpy()
        )

    @property
    def reference_point(self):
        if self.vocs.n_objectives == 1:
            # case for multi-fidelity single objective
            if self.vocs.objectives[self.vocs.objective_names[0]] == "MINIMIZE":
                pt = [-10, 0.0]
            else:
                pt = [10, 0.0]

        else:
            # case for multi-fidelity multi-objective
            pt = []
            for name in self.vocs.objective_names:
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

    def compute_hypervolume(self):
        """ compute hypervolume given data"""

        # create a copy of vocs that includes the fidelity parameter
        vocs_copy = deepcopy(self.vocs)
        vocs_copy.objectives[self.fidelity_parameter] = "MAXIMIZE"

        return compute_hypervolume(self.data, vocs_copy, self.reference_point.numpy())
