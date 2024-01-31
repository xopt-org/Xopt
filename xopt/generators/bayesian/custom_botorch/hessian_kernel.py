import torch
from gpytorch.kernels import RBFKernel


class HessianRBF(RBFKernel):
    def __init__(self, hessian_matrix: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.lower_triangular_decomp = torch.linalg.cholesky(
            torch.linalg.inv(hessian_matrix)
        )

    def forward(self, x1, x2, diag=False, **params):
        x1_star = x1 @ self.lower_triangular_decomp
        x2_star = x2 @ self.lower_triangular_decomp
        return super().forward(x1_star, x2_star, diag, **params)
