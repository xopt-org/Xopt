import torch

from xopt.generators.bayesian.custom_botorch.constraint_transform import Constraint


class TestCustomBotorch:
    def test_constraint_transformation(self):

        constraints = [{0: ["GREATER_THAN", 0.5]}, {0: ["LESS_THAN", 0.5]}]

        for c in constraints:
            test_tensor = torch.rand(2, 3, 1)

            constraint_transform = Constraint(c)
            transformed_tensor = constraint_transform(test_tensor)[0]

            multiplier = 1.0 if c[0][0] == "GREATER_THAN" else -1.0
            assert torch.allclose(
                transformed_tensor, (c[0][1] - test_tensor) * multiplier
            )
            untransformed_tensor = constraint_transform.untransform(transformed_tensor)[
                0
            ]
            assert torch.allclose(untransformed_tensor, test_tensor)

        # apply transform to a subset of the tensor
        for c in constraints:
            test_tensor = torch.rand(2, 3, 5)

            constraint_transform = Constraint(c)
            transformed_tensor = constraint_transform(test_tensor)[0]

            multiplier = 1.0 if c[0][0] == "GREATER_THAN" else -1.0

            true_transformed_tensor = test_tensor.clone()
            true_transformed_tensor[..., 0] = (
                c[0][1] - test_tensor[..., 0]
            ) * multiplier

            assert torch.allclose(transformed_tensor, true_transformed_tensor)
            untransformed_tensor = constraint_transform.untransform(transformed_tensor)[
                0
            ]
            assert torch.allclose(untransformed_tensor, test_tensor)
