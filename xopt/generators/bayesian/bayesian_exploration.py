from typing import Optional

from botorch.acquisition import MCAcquisitionFunction, MCAcquisitionObjective
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from xopt.generators.bayesian.bayesian_generator import (
    BayesianGenerator,
    formatted_base_docstring,
)
from xopt.generators.bayesian.turbo import SafetyTurboController


class BayesianExplorationGenerator(BayesianGenerator):
    """
    Bayesian exploration generator for autonomous characterization.
    """

    name = "bayesian_exploration"
    supports_batch_generation: bool = True
    supports_constraints: bool = True

    __doc__ = "Bayesian exploration generator\n" + formatted_base_docstring()

    _compatible_turbo_controllers = [SafetyTurboController]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.vocs.n_observables == 0:
            raise ValueError(
                "BayesianExplorationGenerator requires at least one observable in the vocs (instead of specifying an objective)."
            )

    def _get_acquisition(self, model: Model) -> MCAcquisitionFunction:
        """
        Get the acquisition function for Bayesian Optimization.

        Parameters
        ----------
        model : Model
            The model used for Bayesian Optimization.

        Returns
        -------
        MCAcquisitionFunction
            The acquisition function for Bayesian Optimization.
        """
        sampler = self._get_sampler(model)
        qPV = qPosteriorVariance(
            model,
            sampler=sampler,
            objective=self._get_objective(),
        )

        return qPV


class qPosteriorVariance(MCAcquisitionFunction):
    """
    q-Posterior Variance acquisition function for Bayesian Optimization.

    Parameters
    ----------
    model : Model
        A fitted model.
    sampler : Optional[MCSampler]
        The sampler used to draw base samples. Defaults to `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
    objective : Optional[MCAcquisitionObjective]
        The MCAcquisitionObjective under which the samples are evaluated. Defaults to `IdentityMCObjective()`.
    posterior_transform : Optional[PosteriorTransform]
        A PosteriorTransform (optional).
    X_pending : Optional[Tensor]
        A `batch_shape x m x d`-dim Tensor of `m` design points that have been submitted for function evaluation but have not yet been evaluated. Concatenated into X upon forward call. Copied and set to have no gradient.

    Methods
    -------
    forward(self, X: Tensor) -> Tensor
        Evaluate qPosteriorVariance on the candidate set `X`.
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
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
        """
        Evaluate qPosteriorVariance on the candidate set `X`.

        Parameters
        ----------
        X : Tensor
            A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design points each.

        Returns
        -------
        Tensor
            A `batch_shape'`-dim Tensor of Posterior Variance values at the given design points `X`, where `batch_shape'` is the broadcasted batch shape of model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        variance_samples = (obj - mean).abs()
        return variance_samples.max(dim=-1)[0].mean(dim=0)
