import torch
from botorch.test_functions import Ackley

from xopt.vocs import VOCS

variables = {f"x{i}": [-5, 10] for i in range(20)}
objectives = {"f": "MINIMIZE"}

vocs = VOCS(variables=variables, objectives=objectives)
fun = Ackley(dim=20, negate=False)


def evaluate_ackley(inputs: dict):
    x = torch.tensor([inputs[f"x{i}"] for i in range(20)])
    return {"f": fun(x).numpy()}
