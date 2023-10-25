import numpy as np
import torch
from botorch.test_functions import Ackley

from xopt.vocs import VOCS

variables = {f"x{i}": [-5, 10] for i in range(20)}
objectives = {"y": "MINIMIZE"}

vocs = VOCS(variables=variables, objectives=objectives)
fun = Ackley(dim=20, negate=False)


def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    dim = 20
    part1 = -a * np.exp(-b / np.sqrt(dim) * np.linalg.norm(x, axis=-1))
    part2 = -(np.exp(np.mean(np.cos(c * x), axis=-1)))
    return part1 + part2 + a + np.e


def evaluate_ackley_np(inputs: dict):
    return {"y": ackley(np.array([inputs[k] for k in sorted(inputs)]))}


def evaluate_ackley(inputs: dict):
    x = torch.tensor([inputs[f"x{i}"] for i in range(20)])
    return {"y": fun(x).numpy()}
