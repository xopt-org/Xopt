import torch

from xopt.vocs import VOCS

variables = {f"x{i}": [-1, 1] for i in range(100)}
objectives = {"f": "MINIMIZE"}

vocs = VOCS(variables=variables, objectives=objectives)


def evaluate_sphere(inputs: dict):
    x = torch.tensor([inputs[f"x{i}"] for i in range(100)])
    return {"f": (x**2).sum().numpy()}
