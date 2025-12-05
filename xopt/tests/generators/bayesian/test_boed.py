import pyro.distributions as dist
import pytest
import torch

from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian.boed import BOEDGenerator
from xopt.numerical_optimizer import GridOptimizer


class TestBOED:
    @pytest.fixture(autouse=True)
    def setup(self):
        ground_truth_x0 = 2.0  # lower edge location
        ground_truth_d = 20  # plateau width
        ground_truth_b = 0.1  # sharpness of the plateau edge

        vocs = VOCS(variables={"x": [0.0, 3.0]}, observables=["y"])
        priors = {
            "x0": dist.Normal(2.0, 1.0),
            "d": dist.Normal(20.0, 1.0),
            "b": dist.Normal(0.1, 0.1),
        }
        generator = BOEDGenerator(
            vocs=vocs,
            model_priors=priors,
            measurement_noise=0.01,
            model_function=self.f,
            numerical_optimizer=GridOptimizer(n_grid_points=10),
        )

        evaluator = Evaluator(
            function=lambda x: {
                "y": float(
                    self.f(
                        torch.tensor(x["x"]),
                        ground_truth_x0,
                        ground_truth_d,
                        ground_truth_b,
                    )
                )
            }
        )

        self.X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

        self.X.grid_evaluate(2)

    def f(self, x, x0, d, b):
        x = x - x0
        return -(torch.tanh(-x / b) + torch.tanh(x / b - d))

    def test_acquisition_function(self):
        boed_generator: BOEDGenerator = self.X.generator
        boed_generator.train_model(self.X.data)
        acquisition_function = boed_generator.get_acquisition(boed_generator.model)

        test_x = torch.linspace(0.0, 3.0, 10).reshape(-1, 1, 1)  # shape (10, 1, 1)
        af_values = acquisition_function(test_x)

        assert af_values.shape == torch.Size([10])

        # try bad input shape
        bad_x = torch.linspace(0.0, 3.0, 10).reshape(-1, 2, 1)  # shape (10, 2, 1)
        with pytest.raises(ValueError):
            af_values = acquisition_function(bad_x)



    def test_full_boed(self):
        self.X.step()

        # eperically this should sample a point near 1.25
        assert torch.allclose(
            torch.tensor(self.X.data["x"].to_numpy()), torch.tensor(1.25), atol=0.5
        )
