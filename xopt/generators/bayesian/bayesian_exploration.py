from typing import Optional

from botorch.acquisition import (
    MCAcquisitionObjective,
    MCAcquisitionFunction,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from xopt import VOCS
from xopt.generators.bayesian.bayesian_generator import MCBayesianGenerator


class BayesianExplorationGenerator(MCBayesianGenerator):
    def __init__(self, vocs: VOCS, **kwargs):
        """
        Generator using UpperConfidenceBound acquisition function

        Parameters
        ----------
        vocs: dict
            Standard vocs for xopt

        **kwargs
            Keyword arguments passed to generator options

        """

        super(BayesianExplorationGenerator, self).__init__(vocs,**kwargs)

    def _get_acquisition(self, model):
        return qPosteriorVariance(model, **self.options.acq.dict())


class qPosteriorVariance(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.
        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.
        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        variance_samples = (obj - mean).abs()
        return variance_samples.max(dim=-1)[0].mean(dim=0)
