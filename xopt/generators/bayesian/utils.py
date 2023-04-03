import torch

from .custom_botorch.constrained_acqusition import FeasibilityObjective

from .objectives import create_constraint_callables


def feasibility(X, model, sampler, vocs, posterior_transform=None):
    constraints = create_constraint_callables(vocs)
    posterior = model.posterior(X=X, posterior_transform=posterior_transform)
    samples = sampler(posterior)
    objective = FeasibilityObjective(constraints)
    return torch.mean(objective(samples, X), dim=0)
