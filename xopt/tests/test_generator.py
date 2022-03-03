import pytest
from ..bayesian.generators.generator import BayesianGenerator
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.exceptions.errors import BotorchError
import torch
from botorch.models import SingleTaskGP
from .evaluators import TNK

import xopt.vocs

def _good_test(model, beta):
    return UpperConfidenceBound(model, beta)


def _bad_test(model):
    return None


class TestGenerator:
    def test_generator(self):
        vocs = xopt.vocs.VOCS.parse_obj(TNK.VOCS)

        train_x = torch.rand(2, len(vocs.variables))
        train_y = torch.ones(2, len(vocs.objectives) +
                                len(vocs.constraints))

        model_1d = SingleTaskGP(train_x, train_y[:, 0].reshape(-1, 1))
        model_md = SingleTaskGP(train_x, train_y)

        beta = 0.01
        ucb = UpperConfidenceBound(model_1d, beta)
        test_x = torch.zeros(1, len(vocs.variables))
        test_ucb_value = ucb(test_x)

        gen = BayesianGenerator(vocs, UpperConfidenceBound,
                                acq_options={'beta': beta})
        assert gen.acq_func(model_1d, **gen.acq_options)(test_x) == test_ucb_value

        gen2 = BayesianGenerator(vocs, _good_test,
                                 acq_options={'beta': beta})

        # test bad model
        with pytest.raises(BotorchError):
            gen2.generate(model_1d)

        # test bad acq function
        with pytest.raises(RuntimeError):
            gen3 = BayesianGenerator(vocs, _bad_test)
            gen3.generate(model_md)
