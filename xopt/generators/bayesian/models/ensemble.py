from botorch.models.model import Model, ModelList
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors.ensemble import EnsemblePosterior

from gpytorch.distributions import MultivariateNormal

import torch
from torch import Tensor
from torch import nn

import numpy as np
from copy import deepcopy

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.utils import get_training_data

from pydantic import Field
from typing import List, Callable, Dict, Union
import pandas as pd


class EnsembleModelConstructor(ModelConstructor):
    """
    Model constructor for ensemble estimates
    """

    name: str = Field("ensemble")
    model: Callable = Field(None, description="Ensemble model")

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.float64,
        device: Union[torch.device, str] = "cpu",
    ):
        # Get training data
        train_X, train_Y, train_Yvar = get_training_data(
            input_names, outcome_names, data
        )
        self.model = self.model.to(dtype)

        # Fit model on training data
        self.model.fit(train_X, train_Y)
        self.model = self.model.to(device)
        self.model.eval()

        # Expected model list
        self.model.models = [self.model]

        return self.model


class MCDropoutMLP(nn.Module):
    """
    Generic multi-layer perceptron model with MC dropout
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate=0.5,
        n_hidden_layers=3,
        hidden_dim=100,
        hidden_activation=nn.ReLU(),
        output_activation=None,
    ):
        super().__init__()

        # Build up model layers
        self.layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            # Input layer or hidden layers
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.layers.append(nn.Dropout(p=dropout_rate))
            self.layers.append(hidden_activation)

        # Output layer
        if n_hidden_layers == 0:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        if output_activation is not None:
            self.layers.append(output_activation)

    def forward(self, input_data, seed=None):
        # Needed so that dropout works
        self.train()
        # Seeding makes dropout self-consistent
        if seed is not None:
            torch.random.manual_seed(seed)
        for layer in self.layers:
            input_data = layer(input_data)

        self.eval()
        return input_data


class FlattenedEnsemblePosterior(EnsemblePosterior):
    """
    Wrapper for EnsemblePosterior to make shapes do similar things
    as GPyTorchPosterior.
    """

    def rsample(self, sample_shape=torch.Size()):
        samples = super().rsample(sample_shape)
        # Flatten (q, m) into a single dimension like GP case
        q, m = samples.shape[-2:]
        return samples.view(*samples.shape[:-2], q * m)

    @property
    def mean(self):
        # Average over ensemble dimension
        return super().mean.mean(dim=1)

    @property
    def variance(self):
        # Average over ensemble dimension
        return super().variance.mean(dim=1)

    @property
    def mvn(self):
        mean = self.mean
        variance = self.variance

        # Assumes diagonal gaussian for simplicity -- not necessarily true, but probably good enough
        covar = torch.diag_embed(variance.squeeze(-1).squeeze(-1))
        mvn = MultivariateNormal(mean.squeeze(-1).squeeze(-1), covariance_matrix=covar)

        return mvn


class MCDropoutModel(Model):
    """
    BoTorch model for MC dropout ensemble
    """

    model: MCDropoutMLP
    num_samples: int
    input_dim: int
    output_dim: int
    dropout_rate: float
    n_hidden_layers: int
    hidden_dim: int

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        num_samples: int = 30,
        n_hidden_layers=3,
        hidden_dim=100,
        hidden_activation=nn.ReLU(),
        output_activation=None,
        observables=["y1"],
        input_bounds=torch.tensor([[-np.pi], [np.pi]]),
        n_epochs=500,
        n_cond_epochs=40,
    ):
        super(MCDropoutModel, self).__init__()
        # Construct model
        self.model = MCDropoutMLP(
            input_dim,
            output_dim,
            dropout_rate=dropout_rate,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

        self.num_samples = num_samples
        self.n_epochs = n_epochs
        self.n_cond_epochs = n_cond_epochs

        self.outcome_transform = Standardize(output_dim)
        self.input_transform = Normalize(input_dim, bounds=input_bounds)
        self.output_indices = None

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        loss_fn=torch.nn.MSELoss(),
        n_epochs=None,
        optim_type=torch.optim.Adam,
        lr=1e-3,
        is_fantasy=False,
    ) -> None:
        if n_epochs is None:
            n_epochs = self.n_epochs

        self.model.train()
        # Seems like this is normalized already in InfoBAX calc?
        if not is_fantasy:
            X_s = self.input_transform(X)
            y_s = self.outcome_transform(y)[0]
        else:
            X_s = X
            y_s = y

        # Generic training loop -- may be worth exposing more/adding more hooks?
        optimizer = optim_type(self.model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            pred = self.model(X_s)
            loss = loss_fn(pred, y_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.train_inputs = (X,)
            self.train_targets = y

    def condition_on_observations(self, X, y):
        fantasies = []
        for i_batch in range(X.shape[0]):
            new_model = deepcopy(self)
            train_X = torch.cat([new_model.train_inputs[0], X[i_batch]], dim=0)
            train_y = torch.cat([new_model.train_targets, y[i_batch]], dim=0)

            new_model.fit(
                train_X, train_y, n_epochs=self.n_cond_epochs, is_fantasy=True
            )

            fantasies.append(new_model)

        return ModelList(*fantasies)

    def subset_output(self, output_indices):
        # Create a deep copy of the model to avoid modifying the original
        new_model = deepcopy(self)
        # Update output indices attribute (see forward)
        new_model.output_indices = output_indices

        return new_model

    def forward(self, X: Tensor) -> Tensor:
        x = self.input_transform(X)

        preds = []
        for i in range(self.num_samples):
            out = self.model(x, seed=i)
            if out.ndim == 2:  # add q dimension
                out = out.unsqueeze(-2)
            elif out.ndim == 1:
                out = out.unsqueeze(-1).unsqueeze(-1)

            out = out.unsqueeze(1)
            preds.append(out)

        samples = torch.cat(preds, dim=1)

        if self.output_indices is not None:
            return samples[..., self.output_indices]
        else:
            return samples

    def posterior(self, X: Tensor, **kwargs):
        samples = self.forward(X)
        if hasattr(self, "outcome_transform"):
            samples = self.outcome_transform.untransform(samples)[0]

        posterior = FlattenedEnsemblePosterior(samples)

        return posterior

    def set_train_data(self, X=None, y=None, strict=False):
        if X is None or y is None:
            return
        self.train_inputs = (X,)
        self.train_targets = y
