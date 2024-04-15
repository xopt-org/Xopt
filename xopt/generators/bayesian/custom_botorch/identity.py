import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class Identity(AcquisitionFunction):
    def __init__(self, model) -> None:
        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.
        Parameters
        ----------
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns
        -------
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """

        return torch.ones(X.shape[:-2])
