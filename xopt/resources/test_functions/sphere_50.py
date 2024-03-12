import torch

from xopt.vocs import VOCS

variables = {f"x{i}": [-1, 1] for i in range(50)}
objectives = {"f": "MINIMIZE"}

vocs = VOCS(variables=variables, objectives=objectives)


def evaluate_sphere(inputs: dict):
    x = torch.tensor([inputs[f"x{i}"] for i in range(50)])
    return {"f": (x**2).sum().numpy()}
