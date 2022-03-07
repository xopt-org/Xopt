import torch.nn
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction, \
    AnalyticAcquisitionFunction
from botorch.utils.objective import apply_constraints_nonnegative_soft
from torch import Tensor
from typing import List, Callable
from torch.nn import Module


class Constrained(AcquisitionFunction):
    def __init__(
            self,
            acq_function: AcquisitionFunction,
            constraints: List[Callable[[Tensor], Tensor]],
            eta: float = 1e-3
    ):
        """
        Wrapper acqusition function to add constraints based on feasibility weighting.
        Note that this requires the base acquisition function to be strictly
        positive. We get around this by applying a SoftPlus transformation
        and then multiplying the transformed acquisition function by the probability.
        This can run into numerical issues if the acqusition function output is strongly
        negative, so use standardization for your outputs if you expect negative
        acquisiton function value.

        Parameters
        ----------
        acq_function : AcquisitionFunction
            The base acquisition function, operating on input tensors
                of feature dimension `d`.

        constraints : dict
            Dictionary containing elements of the form {model_output_index: Callable}
            where the callable transforms the output
        """

        Module.__init__(self)

        self.acq_func = acq_function
        self.constraints = constraints
        self.eta = eta

    def forward(self, X: Tensor) -> Tensor:
        acq_out = self.acq_func(X)
        acq_plus = torch.nn.Softplus()(acq_out)

        posterior = self.acq_func.model.posterior(
            X, posterior_transform=self.acq_func.model.posterior_transform
        )

        if isinstance(self.acq_func, MCAcquisitionFunction):
            samples = self.acq_func.sampler(posterior)
            obj = apply_constraints_nonnegative_soft(
                acq_plus, constraints=self.constraints, samples=samples, eta=self.eta
            )

            return obj
        elif isinstance(self.acq_func, AnalyticAcquisitionFunction):
            mean = posterior.mean
            variance = posterior.variance
