from botorch.acquisition import AcquisitionFunction
from botorch.utils.safe_math import log_softplus
from torch import Tensor
from torch.nn import Module


class LogAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        acq_function: AcquisitionFunction,
    ) -> None:
        Module.__init__(self)
        self.acq_func = acq_function

    def forward(self, X: Tensor) -> Tensor:
        # apply a softplus transform to avoid numerical gradient issues
        return log_softplus(self.acq_func(X), 1e-6)
