from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from botorch.models.transforms.input import Normalize


class CostAwareNormalize(Normalize):
    """
    Wrapper for botorch normalize that ignores the last input value (which should be
    cost)

    """

    def __init__(
        self,
        d: int,
        bounds: Optional[Tensor] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_range: float = 1e-8,
    ) -> None:
        super(CostAwareNormalize, self).__init__(
            d,
            bounds,
            batch_shape,
            transform_on_train,
            transform_on_eval,
            transform_on_fantasize,
            reverse,
        )

    def _transform(self, X: Tensor) -> Tensor:
        # get last values of X
        last_x = X[..., -1]
        trans_x = super()._transform(X)
        trans_x[..., -1] = last_x
        return trans_x

    def _untransform(self, X: Tensor) -> Tensor:
        # get last values of X
        last_x = X[..., -1]
        trans_x = super()._untransform(X)
        trans_x[..., -1] = last_x
        return trans_x
