from abc import ABC, abstractmethod
from typing import List, Dict
from pydantic import BaseModel

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import Module
from gpytorch.mlls import ExactMarginalLogLikelihood

from xopt import Generator, VOCS
from xopt.generators.random import RandomGenerator
from .objectives import create_constrained_mc_objective
from .options import BayesianOptions, AcqOptions, ModelOptions
from .custom_acq.proximal import ProximalAcquisitionFunction


class BayesianGenerator(Generator, ABC):
    _model = None
    _acquisition = None

    def __init__(
            self,
            vocs: VOCS,
            options: BayesianOptions = BayesianOptions(),
            **kwargs
    ):
        super(BayesianGenerator, self).__init__(vocs)

        self.options = options

        if self.options.acq.objective is None:
            self.options.acq.objective = create_constrained_mc_objective(self.vocs)

        if self.options.model.input_transform is None:
            bounds = torch.tensor(vocs.bounds, **self.options.tkwargs)
            self.options.model.input_transform = Normalize(
                len(bounds[0]), bounds=bounds
            )

        if self.options.model.outcome_transform is None:
            self.options.model.outcome_transform = Standardize(1)

        # update generator options w/kwargs
        self.options.update(**kwargs)

    def generate(self, n_candidates: int) -> List[Dict]:

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            gen = RandomGenerator(self.vocs)
            return gen.generate(self.options.n_initial)

        else:
            bounds = torch.tensor(
                self.vocs.bounds, **self.options.tkwargs
            )

            # update internal model with internal data
            model = self.get_model(self.data)

            candidates, _ = optimize_acqf(
                acq_function=self.get_acquisition(model),
                bounds=bounds,
                q=n_candidates,
                **self.options.optim.dict()
            )
            return self.convert_numpy_candidates(candidates.detach().numpy())

    def get_model(self, data: pd.DataFrame = None) -> Module:
        """
        Returns a SingleTaskGP (or ModelList set of independent SingleTaskGPs
        depending on the number of outputs, if data is None

        """

        inputs, outputs = self.get_training_data(data)

        n_outputs = outputs.shape[-1]
        if n_outputs > 1:
            model_list = []
            for i in range(n_outputs):
                m = SingleTaskGP(
                    inputs, outputs[:, i].unsqueeze(1), **self.options.model.dict()
                )
                mll = ExactMarginalLogLikelihood(m.likelihood, m)
                fit_gpytorch_model(mll)
                model_list += [m]

            model = ModelListGP(*model_list)
        else:
            model = SingleTaskGP(inputs, outputs, **self.options.model.dict())

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

        return model

    def get_training_data(self, data: pd.DataFrame = None):
        """overwrite get training data to transform numpy array into tensor"""
        inputs, outputs = super().get_training_data(data)
        return torch.tensor(inputs, **self.options.tkwargs), torch.tensor(
            outputs, **self.options.tkwargs
        )

    def get_acquisition(self, model):
        # add proximal biasing if requested
        if self.options.proximal_lengthscales is not None:
            acq = ProximalAcquisitionFunction(
                self._get_acquisition(model),
                torch.tensor(
                    self.options.proximal_lengthscales,
                    **self.options.tkwargs
                )
            )
        else:
            acq = self._get_acquisition(model)

        return acq

    @abstractmethod
    def _get_acquisition(self, model):
        pass


class MCBayesianGenerator(BayesianGenerator, ABC):
    def __init__(self, vocs, **kwargs):
        super(MCBayesianGenerator, self).__init__(vocs, **kwargs)
