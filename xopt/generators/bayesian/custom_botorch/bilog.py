from typing import List, Optional, Tuple

import torch

from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor


def rms(X, dim=0, keepdim=False):
    return torch.sqrt(torch.sum(X ** 2, dim=dim, keepdim=keepdim) / X.shape[dim])


class Bilog(OutcomeTransform):
    r"""Bilog-transform outcomes.
    The Bilog transform [eriksson2021scalable]_ is useful for modeling outcome
    constraints as it magnifies values near zero and flattens extreme values.
    """

    def __init__(
            self,
            m: int,
            outputs: Optional[List[int]] = None,
            batch_shape: torch.Size = torch.Size(),  # noqa: B008
            min_scale: float = 1e-8,
    ) -> None:
        r"""Bilog-transform outcomes.
        Args:
            outputs: Which of the outputs to Bilog-transform. If omitted, all
                outputs will be transformed.
        """
        super().__init__()

        self._outputs = normalize_indices(outputs, d=m)
        self.register_buffer("scales", torch.zeros(*batch_shape, 1, m))
        self._batch_shape = batch_shape
        self._min_scale = min_scale
        self._m = m

    def forward(
            self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Bilog-transform outcomes.
        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError("wrong batch shape")
            if Y.size(-1) != self._m:
                raise RuntimeError("wrong output dimension")
            scales = rms(Y, dim=-2, keepdim=True)
            scales = scales.where(
                scales >= self._min_scale, torch.full_like(scales, 1.0)
            )
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                scales[..., unused] = 0.0

            self.scales = scales

        Y_tf = Y.sign() * (Y.abs() / self.scales + 1.0).log()
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.
        Args:
            idcs: The output indices to subset the transform to.
        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(self._m, outputs=new_outputs,
                                batch_shape=self._batch_shape,
                                min_scale=self._min_scale
                                )
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform(
            self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform bilog-transformed outcomes
        Args:
            Y: A `batch_shape x n x m`-dim tensor of bilog-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of bilog-transformed
                observation noises associated with the training targets
                (if applicable).
        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        Y_utf = Y.sign() * (Y.abs().exp() - 1.0) * self.scales
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the bilog-transformed posterior.
        Args:
            posterior: A posterior in the bilog-transformed space.
        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Bilog does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.sign() * (x.abs().exp() - 1.0) * self.scales,
        )
