import torch
from ..bayesian.models.models import create_model
import numpy as np
from xopt.vocs import VOCS

class TestModelCreation:
    vocs = VOCS(variables = {'x1': [0, 1],
                          'x2': [0, 1],
                          'x3': [0, 1]} ) 

    def test_create_model(self):
        train_x = torch.rand(5, 3)
        train_y = torch.rand(5, 2)
        train_c = torch.rand(5, 4)

        model = create_model(train_x, train_y, train_c, vocs=self.vocs)

        train_y_nan = train_y.clone()
        train_y_nan[0][1] = np.nan

        model = create_model(train_x, train_y_nan, train_c, vocs=self.vocs)
