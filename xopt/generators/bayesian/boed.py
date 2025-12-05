from typing import Callable, Optional
from matplotlib import pyplot as plt
import pyro
import pyro.distributions as dist
import torch
import logging
from pydantic import Field, PositiveFloat
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

logger = logging.getLogger(__name__)


class BOEDGenerator(BayesianGenerator):
    """
    Bayesian Optimal Experimental Design (BOED) generator for optimization tasks.

    This class implements a generator that utilizes Bayesian methods to design experiments
    optimally. It leverages probabilistic models to select the most informative experiments
    based on prior knowledge and observed data.

    Attributes:
    -----------
    name : str
        The name of the generator.
    model_function : callable
        The probabilistic model used for Bayesian inference.
    model_priors: dict[str, dist.Distribution]
        A dictionary defining the prior distributions for the model parameters.
    measurement_noise: PositiveFloat

    """

    name = "boed"
    model_function: callable
    model_priors: dict[str, dist.Distribution]
    measurement_noise: PositiveFloat

    model: Optional[Callable] = Field(
        None, description="botorch model used by the generator to perform optimization"
    )

    history: list[dict[str, torch.Tensor]] = []

    def get_probabilistic_model_parameters(self):
        """
        Create a dict of parameters for the probabilistic model. Initial
        values of the parameters is taken from the prior distributions.
        """
        params = {}
        for name, distribution in self.model_priors.items():
            if isinstance(distribution, dist.Normal):
                params[f"{name}_loc"] = distribution.loc
                params[f"{name}_scale"] = distribution.scale
            elif isinstance(distribution, dist.Gamma):
                params[f"{name}_concentration"] = distribution.concentration
                params[f"{name}_rate"] = distribution.rate
        return params

    def train_model(self, data):
        """
        Train the probabilistic model using the provided data.

        Parameters:
        -----------
        n_candidates : int
            The number of candidate experiments to generate.
        """
        # start with training a model with historical data
        xs = torch.tensor(data[self.vocs.variable_names].values).float().flatten()
        ys = torch.tensor(data[self.vocs.observable_names].values).float().flatten()

        if len(self.history) == 0:
            self.history.append(self.get_probabilistic_model_parameters())

        current_model = self.get_model(self.history[-1])
        guide = self.get_guide()
        conditioned_model = pyro.condition(current_model, {"y": ys})
        svi = SVI(
            conditioned_model,
            guide,
            Adam({"lr": 0.01}),
            loss=Trace_ELBO(),
        )
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(xs)

        # after training, get the best parameters
        learned_params = {}
        for key in self.history[-1]:
            learned_params[key] = pyro.param(key).detach().clone()
        self.history.append(learned_params)

        for name, value in learned_params.items():
            logger.debug(f"{name}: {value}")

        self.model = self.get_model(learned_params)
        return self.model

    def get_predictive(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return_sites = ["y", "_RETURN"] + list(self.model_priors.keys())
        predictive = Predictive(
            self.model,
            guide=self.get_guide(),
            num_samples=800,
            return_sites=return_sites,
        )
        return predictive

    def _get_acquisition(self, model):
        """
        return a function that computes the log expected information gain

        Parameters:
        -----------
        model : callable
            The probabilistic model for which to compute the acquisition function.
        """

        acquisition = EIGAcquisitionFunction(
            model, list(self.model_priors.keys()), self.measurement_noise
        )

        return acquisition

    def get_model(self, parameters: dict[str, torch.Tensor]):
        # define probabilistic model
        def model(x):
            # dim of -1 in l represents the number of measurements
            # other dims are batch dims

            # sample parameters
            sampled_params = {}
            for name, distribution in self.model_priors.items():
                if isinstance(distribution, dist.Normal):
                    sampled_params[name] = pyro.sample(
                        name,
                        dist.Normal(
                            parameters[f"{name}_loc"],
                            parameters[f"{name}_scale"],
                        ),
                    )
                elif isinstance(distribution, dist.Gamma):
                    sampled_params[name] = pyro.sample(
                        name,
                        dist.Gamma(
                            parameters[f"{name}_concentration"],
                            parameters[f"{name}_rate"],
                        ),
                    )

            # parameters are the same for each measurement
            for key in sampled_params:
                sampled_params[key] = sampled_params[key].unsqueeze(-1)

            # get model output
            output = self.model_function(x, **sampled_params)

            with pyro.plate_stack("data", x.shape[:-1]):
                # observe data
                y = pyro.sample(
                    "y",
                    dist.Normal(output, self.measurement_noise).to_event(1),
                )
                return output

        return model

    def get_guide(self):
        # define a guide function
        def guide(l):
            for name, distribution in self.model_priors.items():
                if isinstance(distribution, dist.Normal):
                    pyro.sample(
                        name,
                        dist.Normal(
                            pyro.param(f"{name}_loc", distribution.loc),
                            pyro.param(
                                f"{name}_scale",
                                distribution.scale,
                                constraint=dist.constraints.positive,
                            ),
                        ),
                    )
                elif isinstance(distribution, dist.Gamma):
                    pyro.sample(
                        name,
                        dist.Gamma(
                            pyro.param(
                                f"{name}_concentration",
                                distribution.concentration,
                                constraint=dist.constraints.positive,
                            ),
                            pyro.param(
                                f"{name}_rate",
                                distribution.rate,
                                constraint=dist.constraints.positive,
                            ),
                        ),
                    )

        return guide


class EIGAcquisitionFunction(torch.nn.Module):
    """
    Acquisition function that computes the expected information gain (EIG)
    for Bayesian Optimal Experimental Design (BOED).

    Parameters
    ----------
    model : Callable
        The probabilistic model used to compute the EIG.
    """

    def __init__(self, model: Callable, eig_keys: list[str], measurement_noise: float):
        super().__init__()
        self.model = model
        self.eig_keys = eig_keys
        self.measurement_noise = measurement_noise

    # define a marginal guide
    def get_marginal_guide(self):
        # define a marginal guide
        def marginal_guide(design, observation_labels, target_labels):
            # This shape allows us to learn a diff parameter for each candidate design l
            # note that this guide samples only the observed sample sites
            # this guide approximates p(y|x) not p(theta|y,x)
            q_y = pyro.param("q_y", torch.zeros(design.shape[-2:]))
            pyro.sample("y", dist.Normal(q_y, self.measurement_noise).to_event(1))

        return marginal_guide

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the EIG acquisition function at the given points.

        Parameters
        ----------
        x : Tensor
            The input points where the acquisition function is evaluated.

        Returns
        -------
        Tensor
            The computed EIG values at the input points.
        """

        pyro.clear_param_store()
        num_steps, start_lr, end_lr = 1000, 0.1, 0.001
        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": start_lr},
                "gamma": (end_lr / start_lr) ** (1 / num_steps),
            }
        )

        eig = marginal_eig(
            self.model,
            x,
            "y",
            self.eig_keys,
            num_samples=100,  # number of mc samples for estimating the EIG
            num_steps=num_steps,
            guide=self.get_marginal_guide(),
            optim=optimizer,
            final_num_samples=10000,  # number of iterations for minimizing the KL divergence of the marginal distribution
        )
        log_eig = eig.log()
        output = torch.nan_to_num(log_eig, nan=float("-inf"))

        return output
