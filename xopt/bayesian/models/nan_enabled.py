from abc import ABC

import torch
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from gpytorch.models import IndependentModelList


class ModelCreationError(Exception):
    pass


class NanEnabledModelListGP(IndependentModelList, ModelListGPyTorchModel):
    def __init__(self, train_x, train_y, **kwargs):
        n_outputs = train_y.shape[-1]

        # check if there are any nans
        # has_nans = torch.any(torch.isnan(train_outputs))
        has_nans = True

        gp_models = []
        if has_nans:
            for ii in range(n_outputs):
                output = train_y[:, ii].flatten()

                nan_state = torch.isnan(output)
                not_nan_idx = torch.nonzero(~nan_state).flatten()

                # remove elements that have nan values
                temp_train_x = train_x[not_nan_idx]
                temp_train_y = output[not_nan_idx].reshape(-1, 1)

                if len(temp_train_y) == 0:
                    print(train_y)
                    raise ModelCreationError('No valid measurements passed to model')

                # create single task model and add to list
                submodel = SingleTaskGP(temp_train_x, temp_train_y, **kwargs)

                mll = ExactMarginalLogLikelihood(submodel.likelihood, submodel)
                fit_gpytorch_model(mll)

                gp_models.append(submodel)

        else:
            model = SingleTaskGP(train_x, train_y, **kwargs)

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            gp_models.append(model)

        self.last_x = train_x[-1]
        super(NanEnabledModelListGP, self).__init__(*gp_models)
