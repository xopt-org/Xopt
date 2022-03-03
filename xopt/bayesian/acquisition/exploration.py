import torch
from botorch.acquisition import MCAcquisitionFunction, AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _construct_dist
from botorch.utils.objective import apply_constraints_nonnegative_soft
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
    convert_to_target_pre_hook,
)
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel


class BayesianExploration(AnalyticAcquisitionFunction):
    r"""Bayesian exploration ie. Proximally Constrained Uncertainty Sampling.
    Computes the analytic expected improvement for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.
    """

    def __init__(self, model, objective_index, constraints=None, sigma=None):
        r"""Analytic Constrained Expected Improvement.
        Args:
            model: A fitted single-outcome model.

            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.objective_index = objective_index
        self.constraints = constraints
        if self.constraints is not None:
            self._preprocess_constraint_bounds(constraints=constraints)

        self.register_forward_pre_hook(convert_to_target_pre_hook)

        # define sigma matrix for proximal term - if not defined set to very large
        if sigma is None:
            if isinstance(self.model, ModelListGPyTorchModel):
                self.sigma = torch.eye(self.model.train_inputs[0][0].shape[-1]) * 1e6
            elif isinstance(self.model, GPyTorchModel):
                self.sigma = torch.eye(self.model.train_inputs[0].shape[-1]) * 1e6
            else:
                raise NotImplementedError(
                    "Get Ryan (rroussel@slac.stanford.edu) to make corrections for "
                    "this type of model in proximal acq, or specify your own sigma "
                    "matrix "
                )
        else:
            self.sigma = sigma

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate Bayesian Exploration on the candidate set X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Bayesian Exploration values at the given
            design points `X`.
        """

        posterior = self.model.posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        if self.constraints is not None:
            # weight the output by feasibility probability
            prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)
            out = sigmas[:, self.objective_index] * prob_feas.flatten()
        else:
            out = sigmas[:, self.objective_index]

        # weight the output by proximity
        prox_weight = self._calculate_proximal(X)
        out = out * prox_weight

        return out

    def _calculate_proximal(self, X):
        # get last observation point
        try:
            last_x = self.model.last_x
        except AttributeError:
            last_x = self.model.train_inputs[0][0][-1]

        # create probability dist
        self.sigma = self.sigma.type(last_x.type())
        d = torch.distributions.MultivariateNormal(last_x, self.sigma)

        # use pdf to calculate the weighting - normalized to 1 at the last point
        norm = torch.exp(d.log_prob(last_x).flatten())
        weight = torch.exp(d.log_prob(X)) / norm
        return weight.flatten()

    def _preprocess_constraint_bounds(self, constraints):
        r"""Set up constraint bounds.
        Args:
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        con_lower, con_lower_inds = [], []
        con_upper, con_upper_inds = [], []
        con_both, con_both_inds = [], []
        con_indices = list(constraints.keys())
        if len(con_indices) == 0:
            raise ValueError("There must be at least one constraint.")
        if self.objective_index in con_indices:
            raise ValueError(
                "Output corresponding to objective should not be a constraint."
            )
        for k in con_indices:
            if constraints[k][0] is not None and constraints[k][1] is not None:
                if constraints[k][1] <= constraints[k][0]:
                    raise ValueError("Upper bound is less than the lower bound.")
                con_both_inds.append(k)
                con_both.append([constraints[k][0], constraints[k][1]])
            elif constraints[k][0] is not None:
                con_lower_inds.append(k)
                con_lower.append(constraints[k][0])
            elif constraints[k][1] is not None:
                con_upper_inds.append(k)
                con_upper.append(constraints[k][1])
        # tensor-based indexing is much faster than list-based advanced indexing
        self.register_buffer("con_lower_inds", torch.tensor(con_lower_inds))
        self.register_buffer("con_upper_inds", torch.tensor(con_upper_inds))
        self.register_buffer("con_both_inds", torch.tensor(con_both_inds))
        # tensor indexing
        self.register_buffer("con_both", torch.tensor(con_both, dtype=torch.float))
        self.register_buffer("con_lower", torch.tensor(con_lower, dtype=torch.float))
        self.register_buffer("con_upper", torch.tensor(con_upper, dtype=torch.float))

    def _compute_prob_feas(self, X, means, sigmas):
        r"""Compute feasibility probability for each batch of X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x m`-dim Tensor of means.
            sigmas: A `(b) x m`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities
        Note: This function does case-work for upper bound, lower bound, and both-sided
        bounds. Another way to do it would be to use 'inf' and -'inf' for the
        one-sided bounds and use the logic for the both-sided case. But this
        causes an issue with autograd since we get 0 * inf.
        TODO: Investigate further.
        """
        output_shape = X.shape[:-2] + torch.Size([1])
        prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)

        if len(self.con_lower_inds) > 0:
            self.con_lower_inds = self.con_lower_inds.to(device=X.device)
            normal_lower = _construct_dist(means, sigmas, self.con_lower_inds)
            prob_l = 1 - normal_lower.cdf(self.con_lower)
            prob_feas = prob_feas.mul(torch.prod(prob_l, dim=-1, keepdim=True))
        if len(self.con_upper_inds) > 0:
            self.con_upper_inds = self.con_upper_inds.to(device=X.device)
            normal_upper = _construct_dist(means, sigmas, self.con_upper_inds)
            prob_u = normal_upper.cdf(self.con_upper)
            prob_feas = prob_feas.mul(torch.prod(prob_u, dim=-1, keepdim=True))
        if len(self.con_both_inds) > 0:
            self.con_both_inds = self.con_both_inds.to(device=X.device)
            normal_both = _construct_dist(means, sigmas, self.con_both_inds)
            prob_u = normal_both.cdf(self.con_both[:, 1])
            prob_l = normal_both.cdf(self.con_both[:, 0])
            prob_feas = prob_feas.mul(torch.prod(prob_u - prob_l, dim=-1, keepdim=True))
        return prob_feas


class qBayesianExploration(MCAcquisitionFunction):
    def __init__(
        self, model, sampler=None, objective=None, X_pending=None, constraints=None
    ):
        super(qBayesianExploration, self).__init__(model, sampler, objective, X_pending)
        self.constraints = constraints

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        r"""Evaluate qBayesianExploration on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """

        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        # get feasibility weights
        # NOTE this might slow down optimization by sending the ones matrix to the gpu every step - should investigate
        feas_weights = apply_constraints_nonnegative_soft(
            torch.ones(obj.shape, device=obj.device, dtype=obj.dtype),
            self.constraints,
            samples,
            eta=1e-3,
        )

        be_mean = obj.mean(dim=0)
        be_samples = (obj - be_mean).abs() * feas_weights
        be_result = be_samples.max(dim=-1)[0].mean(dim=0)
        return be_result
