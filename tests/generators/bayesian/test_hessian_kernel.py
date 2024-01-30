import pandas as pd
import torch
from gpytorch.kernels import ScaleKernel

from xopt.generators.bayesian.custom_botorch.hessian_kernel import HessianRBF
from xopt.generators.bayesian.models.standard import StandardModelConstructor


class TestHessianKernel:
    def test_hessian_kernel(self):
        cov = torch.tensor([[1, -0.8], [-0.8, 1]]).double()
        precision = torch.linalg.inv(cov)
        kernel = ScaleKernel(HessianRBF(precision))

        data = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0], "y": [1.0, 0.0]})

        # test in model constructor
        constructor = StandardModelConstructor(covar_modules={"y": kernel})
        model = constructor.build_model(
            input_names=["x1", "x2"], outcome_names=["y"], data=data
        )

        # test HessianRBF kernel
        test_x = torch.linspace(-5.0, 5.0, 50)
        xx = torch.meshgrid(test_x, test_x)
        pts = torch.vstack([ele.flatten() for ele in xx]).T

        with torch.no_grad():
            post = model.posterior(pts)
            post.mean
