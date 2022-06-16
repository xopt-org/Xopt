from typing import List, Optional, Tuple

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor


class Constraint(OutcomeTransform):
    r"""Constraint-transform outcomes.

    The Bilog transform [eriksson2021scalable]_ is useful for modeling constraining
    functions. It magnifies values near zero and flattens extreme values outside the
    domain [-1,1].
    """

    def __init__(self, constraints: dict, outputs: Optional[List[int]] = None) -> None:
        r"""Bilog-transform outcomes.

        Args:
            outputs: Which of the outputs to Bilog-transform. If omitted, all
                outputs will be transformed.
            constraints: A dictionary of constraints to apply where keys are the
            output indices and values are the constraints to apply (["LESS_THAN",1]).
        """

        super().__init__()
        self._outputs = outputs
        self._constraints = constraints

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
        new_tf = self.__class__(self._constraints, outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def _constraint_transform(self, Y: Tensor, reverse=False) -> Tensor:
        Y_tf = Y.clone()
        for k, val in self._constraints.items():
            if val[0] == "LESS_THAN":
                if reverse:
                    Y_tf[..., k] = Y_tf[..., k] + val[1]
                else:
                    Y_tf[..., k] = Y_tf[..., k] - val[1]
            elif val[0] == "GREATER_THAN":
                Y_tf[..., k] = val[1] - Y_tf[..., k]
            else:
                raise NotImplementedError(f"{val[0]} is not a supported constraint.")
        return Y_tf

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
        # do the transform
        Y_tf = self._constraint_transform(Y)

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
                "Tanh does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform constraint-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of constraint-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of constraint-transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-power transformed outcome observations.
            - The un-power transformed observation noise (if applicable).
        """
        # undo the transform
        Y_utf = self._constraint_transform(Y, reverse=True)

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
        r"""Un-transform the constraint-transformed posterior.

        Args:
            posterior: A posterior in the constraint-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Bilog does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=self._constraint_transform,
        )
