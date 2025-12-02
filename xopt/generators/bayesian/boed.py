import pyro
import pyro.distributions as dist
import torch
from pydantic import PositiveFloat
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator


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
        xs = torch.tensor(self.data[self.vocs.variable_names].values).float().flatten()
        ys = (
            torch.tensor(self.data[self.vocs.observable_names].values).float().flatten()
        )

        if len(self.history) == 0:
            self.history.append(self.get_probabilistic_model_parameters())

        current_model = self.get_model(self.history[-1])
        guide = self.get_guide()
        conditioned_model = pyro.condition(current_model, {"y": ys})
        svi = SVI(
            conditioned_model,
            guide,
            Adam({"lr": 0.005}),
            loss=Trace_ELBO(),
            num_samples=100,
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
            print(f"{name}: {value}")

        predictive = Predictive(
            conditioned_model, 
            guide=guide, 
            num_samples=800,
            return_sites=("y", "_RETURN")
        )

        return predictive, learned_params

    def generate(self, n_candidates) -> list[dict]:
        """
        Generate new candidate experiments based on the current data.

        """
        predictive, learned_params = self.train_model(self.data)

        current_model = self.get_model(learned_params)

        # now use the trained model to select new experiments
        pyro.clear_param_store()

        num_steps, start_lr, end_lr = 1000, 0.1, 0.001
        candidate_designs = torch.linspace(0, 6, 100).unsqueeze(-1)
        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": start_lr},
                "gamma": (end_lr / start_lr) ** (1 / num_steps),
            }
        )

        eig = marginal_eig(
            current_model,
            candidate_designs,
            "y",
            ["x0", "w", "b"],
            num_samples=100,  # number of mc samples for estimating the EIG
            num_steps=num_steps,
            guide=self.get_marginal_guide(),
            optim=optimizer,
            final_num_samples=10000,  # number of iterations for minimizing the KL divergence of the marginal distribution
        )
        best_x = candidate_designs[torch.argmax(eig).detach()]

        return [
            {self.vocs.variable_names[0]: best_x.item()} for _ in range(n_candidates)
        ]

    def get_model(self, parameters: dict[str, torch.Tensor]):
        # define probabilistic model
        def model(x):
            # dim of -1 in l represents the number of measurements
            # other dims are batch dims
            with pyro.plate_stack("data", x.shape[:-1]):
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

                # observe data
                y = pyro.sample(
                    "y",
                    dist.Normal(output, self.measurement_noise).to_event(1),
                )
                return y

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

    def _get_acquisition(self, model):
        pass
