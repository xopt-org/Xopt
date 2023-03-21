from typing import Dict

from xopt import VOCS


def rosenbrock(x):
    """
    Rosenbrock function
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Has a minimum at all ones
        x = (1,1, ... length(x) )

    Parameters
    ----------
    x: array-like of float

    Returns
    -------
    float
        Rosenbrock function value

    """
    n = len(x)

    return sum(
        (100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(n - 1))
    )


def evaluate_rosenbrock(inputs: Dict, label="y", dummy=1) -> Dict[str, float]:
    """
    Evaluate the Rosenbrock function with labeled inputs and outputs.

    Parameters
    ----------
    inputs: Dict[str, float]
        labeled vector of inputs. The labels can be arbitrary.
        labels will be sorted to construct the call to rosenbrock.


    label: str
        Label for the returned function value.
        Default: 'y'

    Returns
    -------
        outputs: dict with a single item label:rosenbrock value


    Example
    -------
         evaluate_rosenbrock( {"x0":1, "x1":1, "x2":1} )
        returns: {'y': 0}

    """

    return {"y": rosenbrock([inputs[k] for k in sorted(inputs)])}


def make_rosenbrock_vocs(n):
    return VOCS(
        variables={f"x{i}": [-2, 2] for i in range(n)}, objectives={"y": "MINIMIZE"}
    )


rosenbrock2_vocs = make_rosenbrock_vocs(2)
