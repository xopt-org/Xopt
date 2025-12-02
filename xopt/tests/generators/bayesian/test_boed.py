import matplotlib.pyplot as plt
import torch
from xopt import Xopt, VOCS, Evaluator
from xopt.generators.bayesian.boed import BOEDGenerator
import pyro.distributions as dist
import matplotlib.pyplot as plt


# define the function
def f(x, x0, w, b):
    return -(
        torch.tanh(-x / b - w / 2 / b + x0 / b) + torch.tanh(x / b - w / 2 / b - x0 / b)
    )


# visualize the ground truth function
ground_truth_x0 = 4.0  # lower edge location
ground_truth_w = 2.5  # plateau width
ground_truth_b = 0.1  # sharpness of the plateau edge
test_x = torch.linspace(0, 6, 100)

fig, ax = plt.subplots()
ax.plot(test_x, f(test_x, x0=ground_truth_x0, w=ground_truth_w, b=ground_truth_b))


vocs = VOCS(variables={"x": [0.0, 6.0]}, observables=["y"])

generator = BOEDGenerator(
    vocs=vocs,
    model_priors={
        "x0": dist.Normal(3.0, 3.0),
        "w": dist.Gamma(2.5, 0.5),
        "b": dist.Gamma(1.0, 1.0),
    },
    measurement_noise=0.05,
    model_function=f,
)

evaluator = Evaluator(
    function=lambda x: {
        "y": float(
            f(torch.tensor(x["x"]), ground_truth_x0, ground_truth_w, ground_truth_b)
        )
    }
)

X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

X.grid_evaluate(5)

for _ in range(5):
    X.step()

X.data.plot.scatter(x="x", y="y", ax=ax, color="red")
plt.show()