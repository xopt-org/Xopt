import torch

from xopt.generators.bayesian.custom_botorch.factor_analysis_distance_rbf import \
    FactorAnalysisDistanceRBF


class TestCustomBotorch:
    def test_factor_analysis_distance_rbf(self):
        # test factor analysis distance kernel
        kernel = FactorAnalysisDistanceRBF(torch.eye(2))
        x1 = torch.eye(2)
        x2 = torch.eye(2)

        result = kernel(x1, x2)

        # compare result with normal RBF Kernel


        assert torch.allclose(result.evaluate_kernel().tensor, gt)
