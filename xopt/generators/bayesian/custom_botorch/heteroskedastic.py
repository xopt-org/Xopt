from typing import NoReturn, Optional

import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    GaussianLikelihood,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.mean import Mean
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.models import ExactGP
from gpytorch.module import Module
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor


class XoptHeteroskedasticSingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Module] = None,
    ) -> None:
        r"""
        Xopt copy of HeteroskedasticSingleTaskGP from botorch which allows for a user
        to specify mean and covariance modules.


        Parameters
        ----------
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        noise_likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
            ),
        )
        noise_model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Yvar,
            likelihood=noise_likelihood,
            outcome_transform=Log(),
            input_transform=input_transform,
        )
        likelihood = _GaussianLikelihoodBase(HeteroskedasticNoise(noise_model))
        # This is hacky -- this class used to inherit from SingleTaskGP, but it
        # shouldn't so this is a quick fix to enable getting rid of that
        # inheritance
        SingleTaskGP.__init__(
            # pyre-fixme[6]: Incompatible parameter type
            self,
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
        )
        self.register_added_loss_term("noise_added_loss")
        self.update_added_loss_term(
            "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    # pyre-fixme[15]: Inconsistent override
    def condition_on_observations(self, *_, **__) -> NoReturn:
        raise NotImplementedError

    # pyre-fixme[15]: Inconsistent override
    def subset_output(self, idcs) -> NoReturn:
        raise NotImplementedError

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
