import pandas as pd
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from matplotlib import pyplot as plt
from torch import Tensor

from xopt.generators.bayesian.models.standard import StandardModelConstructor


class HessianRBF(RBFKernel):
    def __init__(self, hessian_matrix: Tensor, **kwargs):
        super().__init__(**kwargs)
        self.lower_triangular_decomp = torch.linalg.cholesky(
            torch.linalg.inv(hessian_matrix)
        )

    def forward(self, x1, x2, diag=False, **params):
        x1_star = x1 @ self.lower_triangular_decomp
        x2_star = x2 @ self.lower_triangular_decomp
        return super().forward(x1_star, x2_star, diag, **params)


if __name__ == "__main__":
    # train_X = torch.tensor([[0.0, 0.0], [-0.8, 1]]).reshape(2, 2)
    # train_Y = torch.tensor((1.0, 1.0)).reshape(2, 1)

    cov = torch.tensor([[1, -0.8], [-0.8, 1]]).double()
    precision = torch.linalg.inv(cov)
    kernel = ScaleKernel(HessianRBF(precision))

    data = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0], "y": [1.0, 0.0]})

    # test in model constructor
    constructor = StandardModelConstructor(covar_modules={"y": kernel})
    model = constructor.build_model(
        input_names=["x1", "x2"], outcome_names=["y"], data=data
    )

    # model = SingleTaskGP(train_X, train_Y, covar_module=kernel)

    # test HessianRBF kernel
    test_x = torch.linspace(-5.0, 5.0, 50)
    xx = torch.meshgrid(test_x, test_x)
    pts = torch.vstack([ele.flatten() for ele in xx]).T

    with torch.no_grad():
        post = model.posterior(pts)
        mean = post.mean

    fig, ax = plt.subplots()
    c = ax.pcolor(*xx, mean.reshape(50, 50))
    fig.colorbar(c)
    plt.show()
