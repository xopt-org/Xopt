"""
Meta-LSQ surrogate models for Xopt's Bayesian generators.

Meta-LSQ [1]_ is a surrogate for a family of related objectives, for example one
machine or simulator under changing conditions or targets. Each task is fitted by one
regularised least-squares solve over a frozen basis of N random Fourier features,

    u(x) = (1 / sqrt(N)) * sum_j beta_j sin(sigma * w_j^T x + b_j),

    beta* = argmin ||H beta - y||^2 + w * ||G beta - dy||^2 + mu * ||beta||^2,

where ``H`` holds the features at the data and ``G`` their gradients. The gradient of
a sinusoid is again a sinusoid, so gradient observations ``dy`` enter the same solve
in closed form [2]_. Only the bandwidth ``sigma``, the ridge ``mu`` and the gradient
weight ``w`` are learned, once, across tasks of the family (``MetaLSQ.meta_fit``).
A new task is then conditioned on its data without any hyperparameter fitting
(``MetaLSQModel``), and the model returns the Gaussian posterior of the equivalent
Bayesian linear regression.

``MetaLSQModelConstructor`` plugs meta-trained models into Xopt's Bayesian generators.

References
----------
.. [1] A. Sulc. Meta-LSQ: A Closed-Form Sobolev Inner Solver for Few-Shot
   Meta-Learning. Northern Lights Deep Learning Conference, PMLR 352, 2027.
.. [2] A. Sulc. FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact
   Analytical Derivatives. ICLR 2026 AI & PDE workshop, arXiv:2602.10541.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import FantasizeMixin, Model, ModelList
from botorch.models.transforms import Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import DenseLinearOperator
from pydantic import ConfigDict, Field, field_validator
from torch import nn, Tensor

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.models.standard import (
    _get_non_degenerate_normalize_bounds,
)
from xopt.pydantic import decode_torch_module


@dataclass
class MetaLSQTask:
    r"""One task of a meta-training set.

    Attributes:
        train_X: A ``n x d`` tensor of support inputs.
        train_Y: A ``n x m`` tensor of support observations.
        test_X: A ``n' x d`` tensor of query inputs.
        test_Y: A ``n' x m`` tensor of query observations.
        train_dY: An optional ``n x m x d`` tensor with the gradient of each
            output at the support inputs.
    """

    train_X: Tensor
    train_Y: Tensor
    test_X: Tensor
    test_Y: Tensor
    train_dY: Tensor | None = None


def _ridge_solve(A: Tensor, t: Tensor, mu: Tensor) -> Tensor:
    r"""Solve ``argmin ||A beta - t||^2 + mu ||beta||^2`` (differentiable).

    Uses the ``n x n`` dual system when ``A`` has fewer rows than columns, the
    usual few-shot case, and the ``N x N`` primal system otherwise.
    """
    n, N = A.shape[-2:]
    if n < N:
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        return A.mT @ torch.linalg.solve(A @ A.mT + mu * eye, t)
    eye = torch.eye(N, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A.mT @ A + mu * eye, A.mT @ t)


class MetaLSQ(nn.Module):
    r"""Meta-learner of the bandwidth, ridge and gradient weight of Meta-LSQ.

    Its only trainable parameters are ``log_sigma``, ``log_mu`` and ``log_w``
    (plus the prior-mean coefficients if ``prior_mean=True``). The frequencies and
    phases are buffers drawn once from ``seed``.

    Example:
        >>> meta = MetaLSQ(dim=2, num_outputs=1)
        >>> meta.meta_fit(tasks, num_steps=2000)  # list of MetaLSQTask, or a sampler
        >>> model = meta.condition(train_X, train_Y, train_dY)
        >>> acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    """

    def __init__(
        self,
        dim: int,
        num_outputs: int = 1,
        num_features: int = 200,
        use_gradients: bool = True,
        ard: bool = False,
        prior_mean: bool = False,
        sigma: float = 1.0,
        mu: float = 1e-4,
        w: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        r"""Initialize the Meta-LSQ meta-learner.

        Args:
            dim: The input dimension ``d``.
            num_outputs: The number of outputs ``m``. All outputs share the basis,
                the scalars and one factorisation.
            num_features: The number of random Fourier features ``N``.
            use_gradients: If True, gradient observations can be used and the
                gradient weight ``w`` is learned. If False, ``w`` is frozen and
                gradient observations are rejected.
            ard: If True, learn one bandwidth per input dimension instead of one
                shared bandwidth.
            prior_mean: If True, meta-learn coefficients ``beta_0`` that the ridge
                shrinks towards instead of zero, i.e. a prior mean shared by all
                tasks. Useful when the task family has a non-zero mean.
            sigma: The initial bandwidth (frequency scale).
            mu: The initial ridge.
            w: The initial gradient weight.
            seed: The seed of the frozen frequencies and phases.
            dtype: The dtype of all tensors.
            device: The device of all tensors.
        """
        super().__init__()
        if dim < 1 or num_outputs < 1 or num_features < 1:
            raise ValueError("dim, num_outputs and num_features must be positive.")
        tkwargs = {"dtype": dtype, "device": device}
        gen = torch.Generator().manual_seed(seed)
        W_hat = torch.randn(dim, num_features, generator=gen, dtype=dtype)
        b = torch.rand(num_features, generator=gen, dtype=dtype) * 2 * math.pi
        self.register_buffer("W_hat", W_hat.to(device))
        self.register_buffer("b", b.to(device))
        sigma_shape = (dim,) if ard else ()
        self.log_sigma = nn.Parameter(
            torch.full(sigma_shape, math.log(sigma), **tkwargs)
        )
        self.log_mu = nn.Parameter(torch.tensor(math.log(mu), **tkwargs))
        self.log_w = nn.Parameter(
            torch.tensor(math.log(w), **tkwargs),
            requires_grad=use_gradients,
        )
        if prior_mean:
            self.beta0 = nn.Parameter(torch.zeros(num_features, num_outputs, **tkwargs))
        else:
            self.register_parameter("beta0", None)
        self.dim = dim
        self.num_outputs = num_outputs
        self.num_features = num_features
        self.use_gradients = use_gradients

    @property
    def sigma(self) -> Tensor:
        r"""The bandwidth, a scalar or a ``d``-dim tensor with ``ard=True``."""
        return self.log_sigma.exp()

    @property
    def mu(self) -> Tensor:
        r"""The ridge."""
        return self.log_mu.exp()

    @property
    def w(self) -> Tensor:
        r"""The weight of the gradient rows."""
        return self.log_w.exp()

    @property
    def frequencies(self) -> Tensor:
        r"""The ``d x N`` frequencies, ``sigma * W_hat`` (row-wise with ARD)."""
        return self.sigma.unsqueeze(-1) * self.W_hat

    def design(
        self,
        X: Tensor,
        Y: Tensor,
        dY: Tensor | None = None,
        W: Tensor | None = None,
        sqrt_w: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Build the stacked design ``A`` and target ``t`` of one or more tasks.

        Args:
            X: A ``batch_shape x n x d`` tensor of inputs.
            Y: A ``batch_shape x n x m`` tensor of observations.
            dY: An optional ``batch_shape x n x m x d`` tensor of gradients.
            W: The frequencies to use (defaults to the current ones).
            sqrt_w: The square root of the gradient weight (defaults to the current
                one).

        Returns:
            A ``batch_shape x r x N`` design and a ``batch_shape x r x m`` target,
            with ``r = n`` rows for values and ``n * d`` more for gradients.
        """
        _check(X, Y, dY, self.dim, self.num_outputs, self.use_gradients)
        W = self.frequencies if W is None else W
        A = _features(X, W, self.b)
        t = Y
        if dY is not None:
            sqrt_w = self.w.sqrt() if sqrt_w is None else sqrt_w
            A_g, t_g = _gradient_rows(X, dY, W, self.b, sqrt_w)
            A = torch.cat([A, A_g], dim=-2)
            t = torch.cat([t, t_g], dim=-2)
        return A, t

    def adapt(self, X: Tensor, Y: Tensor, dY: Tensor | None = None) -> Tensor:
        r"""Fit one task by a single ridge solve, differentiable in the scalars.

        Args:
            X: A ``batch_shape x n x d`` tensor of inputs.
            Y: A ``batch_shape x n x m`` tensor of observations.
            dY: An optional ``batch_shape x n x m x d`` tensor of gradients.

        Returns:
            The ``batch_shape x N x m`` coefficients.
        """
        A, t = self.design(X, Y, dY)
        if self.beta0 is None:
            return _ridge_solve(A, t, self.mu)
        return self.beta0 + _ridge_solve(A, t - A @ self.beta0, self.mu)

    def predict(self, X: Tensor, beta: Tensor) -> Tensor:
        r"""Evaluate the fitted model with coefficients ``beta`` at ``X``.

        Args:
            X: A ``batch_shape x q x d`` tensor of inputs.
            beta: The ``batch_shape x N x m`` coefficients from ``adapt``.

        Returns:
            A ``batch_shape x q x m`` tensor of predictions.
        """
        return _features(X, self.frequencies, self.b) @ beta

    def query_loss(self, tasks: Sequence[MetaLSQTask]) -> Tensor:
        r"""The default meta-objective: query MSE after adaptation, averaged over
        tasks and outputs.

        Args:
            tasks: The tasks of one meta-batch.

        Returns:
            A scalar tensor.
        """
        losses = []
        for task in tasks:
            beta = self.adapt(task.train_X, task.train_Y, task.train_dY)
            losses.append((self.predict(task.test_X, beta) - task.test_Y).pow(2).mean())
        return torch.stack(losses).mean()

    def meta_fit(
        self,
        tasks: (
            Sequence[MetaLSQTask]
            | Callable[[int, torch.Generator], Sequence[MetaLSQTask]]
        ),
        num_steps: int = 2000,
        batch_size: int = 4,
        lr: float = 5e-3,
        loss_fn: Callable[[MetaLSQ, Sequence[MetaLSQTask]], Tensor] | None = None,
        seed: int = 0,
        value_warmup_steps: int = 0,
    ) -> list[float]:
        r"""Meta-train the scalars with Adam, differentiating through the solve.

        ``torch.linalg.solve`` is differentiated implicitly, so there is no
        unrolled inner loop and no second-order graph.

        Args:
            tasks: Either a sequence of tasks, from which meta-batches are drawn
                with replacement, or a callable ``tasks(batch_size, generator)``
                that returns a fresh meta-batch.
            num_steps: The number of Adam steps.
            batch_size: The number of tasks per step.
            lr: The learning rate.
            loss_fn: An optional ``loss_fn(meta_model, tasks)`` replacing the query
                MSE.
            seed: The seed of the meta-batch draws.
            value_warmup_steps: For this many initial steps, gradient observations
                are left out of the support fits, so the bandwidth and ridge are
                first learned from values alone. When gradients are much larger
                than values (steep objectives), training with them from the start
                can settle on a small bandwidth that fits neither. A warm-up of
                about half the steps avoided that in the community notebook.

        Returns:
            The list of per-step losses.
        """
        gen = torch.Generator().manual_seed(seed)
        if callable(tasks):
            sampler = tasks
        else:
            pool = list(tasks)
            if not pool:
                raise ValueError("`tasks` is empty.")

            def sampler(k: int, g: torch.Generator) -> list[MetaLSQTask]:
                idx = torch.randint(len(pool), (k,), generator=g).tolist()
                return [pool[i] for i in idx]

        loss_fn = loss_fn or (lambda model, batch: model.query_loss(batch))
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        history = []
        self.train()
        for step in range(num_steps):
            optimizer.zero_grad()
            batch = sampler(batch_size, gen)
            if step < value_warmup_steps:
                batch = [replace(task, train_dY=None) for task in batch]
            loss = loss_fn(self, batch)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def condition(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_dY: Tensor | None = None,
        noise_var: float | Tensor | None = None,
    ) -> MetaLSQModel:
        r"""Condition on one task's data and return a BoTorch model.

        Args:
            train_X: A ``batch_shape x n x d`` tensor of training inputs.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            train_dY: An optional ``batch_shape x n x m x d`` tensor of gradients.
            noise_var: An optional value noise variance. If omitted, it is
                estimated from the data (see ``MetaLSQModel``).

        Returns:
            A ``MetaLSQModel`` with the current scalars frozen.
        """
        return MetaLSQModel(
            meta_model=self,
            train_X=train_X,
            train_Y=train_Y,
            train_dY=train_dY,
            noise_var=noise_var,
        )

    def extra_repr(self) -> str:
        s = f"sigma={_fmt(self.sigma)}, mu={self.mu.item():.3g}"
        if self.use_gradients:
            s += f", w={_fmt(self.w)}"
        return s + f", N={self.num_features}, d={self.dim}, m={self.num_outputs}"


class MetaLSQModel(Model, FantasizeMixin):
    r"""The posterior of Meta-LSQ on one task, with meta-learned scalars frozen.

    Conditioning is one Cholesky factorisation of an ``N x N`` matrix; there are
    no hyperparameters to fit. The posterior over the coefficients is
    ``N(beta*, s^2 (A^T A + mu I)^{-1})`` with ``A`` the stacked design (value
    rows, and gradient rows weighted by ``sqrt(w)``). Unless ``noise_var`` is
    given, ``s^2`` is its maximum-likelihood estimate,
    ``(||t - A beta*||^2 + mu ||beta* - beta_0||^2) / r`` over the ``r`` rows,
    one per output.

    ``condition_on_observations`` (and hence ``fantasize``) appends value rows,
    and gradient rows if ``dY`` is passed, and refactorises.
    """

    likelihood = None  # read by FantasizeMixin.fantasize; the noise is homoscedastic

    def __init__(
        self,
        meta_model: MetaLSQ,
        train_X: Tensor,
        train_Y: Tensor,
        train_dY: Tensor | None = None,
        noise_var: float | Tensor | None = None,
    ) -> None:
        r"""Initialize the model on one task's data.

        Args:
            meta_model: A (meta-trained) ``MetaLSQ``. Its scalars are copied, so
                later training of ``meta_model`` does not change this model.
            train_X: A ``batch_shape x n x d`` tensor of training inputs.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            train_dY: An optional ``batch_shape x n x m x d`` tensor with the
                gradient of each output at the training inputs.
            noise_var: An optional value noise variance (a float, or a tensor
                broadcastable to ``batch_shape x m``). If omitted, it is estimated
                from the data.
        """
        super().__init__()
        if train_X.shape[-2] == 0 and noise_var is None:
            raise ValueError("Pass `noise_var` to condition on an empty data set.")
        self.dim = meta_model.dim
        self._num_outputs = meta_model.num_outputs
        self.use_gradients = meta_model.use_gradients
        with torch.no_grad():
            self.register_buffer("W", meta_model.frequencies.detach().clone())
            self.register_buffer("b", meta_model.b.detach().clone())
            self.register_buffer("mu", meta_model.mu.detach().clone())
            self.register_buffer(
                "sqrt_w",
                meta_model.w.detach().sqrt().clone()
                if meta_model.use_gradients
                else None,
            )
            self.register_buffer(
                "beta0",
                None if meta_model.beta0 is None else meta_model.beta0.detach().clone(),
            )
        self.noise_var = noise_var
        A, t = self._design(train_X, train_Y, train_dY)
        self._set_system(A, t)

    def _design(self, X: Tensor, Y: Tensor, dY: Tensor | None) -> tuple[Tensor, Tensor]:
        _check(X, Y, dY, self.dim, self._num_outputs, self.use_gradients)
        A = _features(X, self.W, self.b)
        t = Y
        if dY is not None:
            A_g, t_g = _gradient_rows(X, dY, self.W, self.b, self.sqrt_w)
            A = torch.cat([A, A_g], dim=-2)
            t = torch.cat([t, t_g], dim=-2)
        batch = torch.broadcast_shapes(A.shape[:-2], t.shape[:-2])
        return A.expand(*batch, *A.shape[-2:]), t.expand(*batch, *t.shape[-2:])

    def _set_system(self, A: Tensor, t: Tensor) -> None:
        r"""Factorise ``A^T A + mu I`` and compute the coefficients and noise."""
        N = A.shape[-1]
        eye = torch.eye(N, dtype=A.dtype, device=A.device)
        L = torch.linalg.cholesky(A.mT @ A + self.mu * eye)
        r = t if self.beta0 is None else t - A @ self.beta0
        delta = torch.cholesky_solve(A.mT @ r, L)  # beta* - beta_0
        if self.noise_var is None:
            rows = A.shape[-2]
            sse = (r - A @ delta).pow(2).sum(-2) + self.mu * delta.pow(2).sum(-2)
            noise_var = sse / rows
        else:
            noise_var = torch.as_tensor(
                self.noise_var, dtype=A.dtype, device=A.device
            ).expand(*A.shape[:-2], t.shape[-1])
        self.register_buffer("_A", A)
        self.register_buffer("_t", t)
        self.register_buffer("_L", L)
        self.register_buffer(
            "_beta", delta if self.beta0 is None else delta + self.beta0
        )
        self.register_buffer("_noise_var", noise_var.clamp_min(1e-12))

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model."""
        return self._A.shape[:-2]

    @property
    def noise_variance(self) -> Tensor:
        r"""The ``batch_shape x m`` value noise variance in use."""
        return self._noise_var

    @property
    def coefficients(self) -> Tensor:
        r"""The ``batch_shape x N x m`` posterior mean of the coefficients."""
        return self._beta

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A ``(batch_shape) x q x d``-dim tensor of inputs.
            output_indices: An optional list of output indices.
            observation_noise: If True, add the model's noise variance. If a
                tensor (``batch_shape x q x m`` or ``batch_shape x 1 x m``), add it
                as the observation noise variance.
            posterior_transform: An optional PosteriorTransform.
            kwargs: Ignored, for compatibility.

        Returns:
            A ``GPyTorchPosterior`` with a ``MultivariateNormal`` (one output) or a
            ``MultitaskMultivariateNormal`` (several outputs).
        """
        self.eval()
        X = self.transform_inputs(X)
        Phi = _features(X, self.W, self.b)  # b x q x N
        mean = Phi @ self._beta  # b x q x m
        V = torch.linalg.solve_triangular(self._L, Phi.mT, upper=False)  # b x N x q
        K = V.mT @ V  # b x q x q
        noise_var = self._noise_var.unsqueeze(-1).unsqueeze(-1)  # batch x m x 1 x 1
        covar = K.unsqueeze(-3) * noise_var  # b x m x q x q
        q = X.shape[-2]
        if isinstance(observation_noise, Tensor):
            noise = observation_noise.transpose(-1, -2)  # ... x m x (q or 1)
            noise = noise.expand(*noise.shape[:-1], q)
            covar = covar + torch.diag_embed(noise)
        elif observation_noise:
            eye = torch.eye(q, dtype=X.dtype, device=X.device)
            covar = covar + noise_var * eye
        if output_indices is not None:
            mean = mean[..., output_indices]
            covar = covar[..., output_indices, :, :]
        # The covariance has rank at most N and is often numerically singular, so
        # it is passed lazily: GPyTorch then factorises it with jitter on demand.
        mvns = [
            MultivariateNormal(mean[..., i], DenseLinearOperator(covar[..., i, :, :]))
            for i in range(mean.shape[-1])
        ]
        if len(mvns) == 1:
            mvn = mvns[0]
        else:
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns)
        posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, dY: Tensor | None = None, **kwargs: Any
    ) -> MetaLSQModel:
        r"""Condition the model on new observations.

        Args:
            X: A ``batch_shape x n' x d``-dim tensor of new inputs.
            Y: A ``batch_shape' x n' x m``-dim tensor of new observations, where
                ``batch_shape'`` broadcasts with ``batch_shape`` and the model's
                batch shape (e.g. fantasies).
            dY: An optional ``batch_shape' x n' x m x d`` tensor of new gradients.
            kwargs: Ignored (e.g. ``noise`` passed by ``fantasize``; the noise
                variance is re-estimated unless the model was built with a
                fixed ``noise_var``).

        Returns:
            A new ``MetaLSQModel`` conditioned on the old and new data.
        """
        A_new, t_new = self._design(X, Y, dY)
        batch = torch.broadcast_shapes(self.batch_shape, A_new.shape[:-2])
        A = torch.cat(
            [
                self._A.expand(*batch, *self._A.shape[-2:]),
                A_new.expand(*batch, *A_new.shape[-2:]),
            ],
            dim=-2,
        )
        t = torch.cat(
            [
                self._t.expand(*batch, *self._t.shape[-2:]),
                t_new.expand(*batch, *t_new.shape[-2:]),
            ],
            dim=-2,
        )
        new = copy.copy(self)
        new._modules = dict(self._modules)
        new._buffers = dict(self._buffers)
        new._set_system(A, t)
        return new

    def subset_output(self, idcs: list[int]) -> MetaLSQModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A ``MetaLSQModel`` with only the given outputs.
        """
        new = copy.copy(self)
        new._modules = dict(self._modules)
        new._buffers = dict(self._buffers)
        if self.beta0 is not None:
            new._buffers["beta0"] = self.beta0[:, idcs]
        if isinstance(self.noise_var, Tensor) and self.noise_var.dim() > 0:
            new.noise_var = self.noise_var[..., idcs]
        new._num_outputs = len(idcs)
        new._set_system(self._A, self._t[..., idcs])
        return new


def _fmt(v: Tensor) -> str:
    v = v.detach()
    if v.dim() == 0:
        return f"{v.item():.4g}"
    return "[" + ", ".join(f"{x:.3g}" for x in v.tolist()) + "]"


def _check(
    X: Tensor, Y: Tensor, dY: Tensor | None, dim: int, m: int, use_gradients: bool
) -> None:
    r"""Validate the shapes of inputs, observations and gradients."""
    if X.shape[-1] != dim:
        raise ValueError(f"Expected inputs of dimension {dim}, got {X.shape}.")
    if Y.dim() < 2 or Y.shape[-1] != m:
        raise ValueError(f"Expected observations of shape `n x {m}`, got {Y.shape}.")
    if Y.shape[-2] != X.shape[-2]:
        raise ValueError(
            f"Inputs and observations have different numbers of points: "
            f"{X.shape} and {Y.shape}."
        )
    if dY is not None:
        if not use_gradients:
            raise ValueError(
                "Gradient observations were given, but `use_gradients=False`."
            )
        if dY.shape[-3:] != (X.shape[-2], m, dim):
            raise ValueError(
                f"Expected gradients of shape `n x {m} x {dim}`, got {dY.shape}."
            )


def _features(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    r"""Value features ``sin(X W + b) / sqrt(N)``, ``batch x n x N``."""
    return torch.sin(X @ W + b) / math.sqrt(W.shape[-1])


def _gradient_rows(
    X: Tensor, dY: Tensor, W: Tensor, b: Tensor, sqrt_w: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Gradient rows of the design and their targets, weighted by ``sqrt(w)``
    and ordered by point, then input dimension."""
    G = sqrt_w * _gradient_features(X, W, b)  # batch x n x d x N
    t = sqrt_w * dY.transpose(-1, -2)  # batch x n x d x m
    return G.flatten(-3, -2), t.flatten(-3, -2)


def _gradient_features(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    r"""Features of each partial derivative, ``batch x n x d x N``."""
    c = torch.cos(X @ W + b) / math.sqrt(W.shape[-1])
    return c.unsqueeze(-2) * W


class MetaLSQModelConstructor(ModelConstructor):
    """
    Build Meta-LSQ models, one per outcome, from meta-trained ``MetaLSQ`` modules.

    Inputs are mapped to the unit cube with the variable bounds before they reach the
    model, so meta-train on inputs scaled the same way. Outcomes are not
    standardised, because the meta-learned scalars refer to the scale of the
    meta-training data. There is no hyperparameter fitting: building a model is one
    Cholesky factorisation per outcome.

    Gradient observations: if ``use_gradients`` is True, the outcome's ``MetaLSQ`` was
    built with ``use_gradients=True`` and the data has a column
    ``<outcome>_grad_<variable>`` for every input variable, these columns are used as
    the gradient of the outcome, in the original (unscaled) variable units. Rows with
    a missing gradient contribute their value only. ``<outcome>_var`` columns are
    ignored: the noise level is estimated from the data or set with ``noise_var``.

    Attributes
    ----------
    name : str
        The name of the model constructor (frozen).
    meta_models : Dict[str, MetaLSQ]
        A meta-trained ``MetaLSQ`` with ``num_outputs=1`` for every outcome, keyed by
        outcome name. Values may also be file paths or ``base64:`` strings of saved
        modules.
    use_gradients : bool
        Use gradient columns when they are present.
    noise_var : Optional[float]
        A fixed noise variance for the outcome values. If None, it is estimated from
        the data of each outcome.
    """

    name: str = Field("meta_lsq", frozen=True)
    meta_models: Dict[str, MetaLSQ] = Field(
        description="meta-trained MetaLSQ modules keyed by outcome name"
    )
    use_gradients: bool = Field(
        True, description="use <outcome>_grad_<variable> columns when present"
    )
    noise_var: Optional[float] = Field(
        None,
        description="fixed noise variance of the outcome values, estimated if None",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("meta_models", mode="before")
    @classmethod
    def validate_meta_models(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("meta_models must be a dict keyed by outcome name")
        value = dict(cast(dict[str, Any], value))
        for key, val in value.items():
            if isinstance(val, str):
                if val.startswith("base64:"):
                    value[key] = decode_torch_module(val)
                elif os.path.exists(val):
                    value[key] = torch.load(val, weights_only=False)
                else:
                    raise ValueError(f"cannot load a MetaLSQ module from {val}")
            if not isinstance(value[key], MetaLSQ):
                raise ValueError(
                    f"meta_models[{key!r}] must be a MetaLSQ, got {type(value[key])}"
                )
            if value[key].num_outputs != 1:
                raise ValueError(f"meta_models[{key!r}] must have num_outputs=1")
        return value

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelList:
        """
        Condition the meta-trained models on the data, one per outcome.

        Parameters
        ----------
        input_names : List[str]
            Names of input variables.
        outcome_names : List[str]
            Names of outcome variables.
        data : pd.DataFrame
            Data used for conditioning the models.
        input_bounds : Dict[str, List]
            Bounds of the input variables, used to map inputs to the unit cube.
        dtype : torch.dtype, optional
            Data type for the model (default is torch.double).
        device : Union[torch.device, str], optional
            Device on which to perform computations (default is "cpu").

        Returns
        -------
        ModelList
            A ``ModelList`` of ``MetaLSQModel``, one per outcome.
        """
        if input_bounds is None:
            raise ValueError("MetaLSQModelConstructor needs input bounds")
        tkwargs = {"dtype": dtype, "device": device}
        bounds = _get_non_degenerate_normalize_bounds(
            input_names, input_bounds, tkwargs
        )
        lower, width = bounds[0], bounds[1] - bounds[0]
        models = []
        for outcome_name in outcome_names:
            if outcome_name not in self.meta_models:
                raise ValueError(
                    f"no meta-trained model for outcome {outcome_name!r}; when "
                    "loading from a file, save with serialize_torch=True so that "
                    "the meta-trained models are stored"
                )
            meta = copy.deepcopy(self.meta_models[outcome_name]).to(**tkwargs)
            if meta.dim != len(input_names):
                raise ValueError(
                    f"meta_models[{outcome_name!r}] has {meta.dim} inputs, the "
                    f"problem has {len(input_names)}"
                )
            X, Y, G = self._get_data(input_names, outcome_name, data, meta, tkwargs)
            if X.shape[0] == 0:
                raise ValueError("no data found to build model!")
            U = (X - lower) / width
            if G is None:
                model = meta.condition(U, Y, noise_var=self.noise_var)
            else:
                # dy/du = dy/dx * (upper - lower) for u the unit-cube input
                dY = (G * width).unsqueeze(-2)
                has_grad = ~torch.isnan(G).any(dim=-1)
                if has_grad.any():
                    model = meta.condition(
                        U[has_grad], Y[has_grad], dY[has_grad], noise_var=self.noise_var
                    )
                    if (~has_grad).any():
                        model = model.condition_on_observations(
                            U[~has_grad], Y[~has_grad]
                        )
                else:
                    model = meta.condition(U, Y, noise_var=self.noise_var)
            model.input_transform = Normalize(len(input_names), bounds=bounds)
            models.append(model)
        return ModelList(*models)

    def _get_data(
        self,
        input_names: List[str],
        outcome_name: str,
        data: pd.DataFrame,
        meta: MetaLSQ,
        tkwargs: dict,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Rows with all inputs and the outcome present; gradients if used."""
        grad_cols = [f"{outcome_name}_grad_{name}" for name in input_names]
        use_grads = (
            self.use_gradients
            and meta.use_gradients
            and all(col in data for col in grad_cols)
        )
        valid = data[input_names + [outcome_name]].notnull().all(axis=1)
        rows = data[valid]
        X = torch.tensor(rows[input_names].to_numpy(dtype="double"), **tkwargs)
        Y = torch.tensor(rows[[outcome_name]].to_numpy(dtype="double"), **tkwargs)
        if not use_grads:
            return X, Y, None
        G = torch.tensor(rows[grad_cols].to_numpy(dtype="double"), **tkwargs)
        return X, Y, G
