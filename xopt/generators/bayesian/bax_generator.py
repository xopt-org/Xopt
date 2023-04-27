# +
import logging
import time
from typing import Callable, Dict, List

from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_truncated_normal_perturbations

from xopt.generators.bayesian.bax.acquisition import ExpectedInformationGain

from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.objectives import create_mc_objective
from xopt.generators.bayesian.options import BayesianOptions
from xopt.vocs import VOCS

logger = logging.getLogger()


class BAXGenerator(BayesianGenerator):
    alias = "BAX"

    def __init__(
        self,
        vocs: VOCS,
        meas_param: str,
        algo_class: Callable,
        algo_kwargs: Dict,
        options: BayesianOptions = None,
    ):
        """
        Generator using Bayesian Algorithm Execution
        via Expected Information Gain acquisition function.

        Parameters
        ----------
        vocs: dict
            Standard vocs dictionary for xopt

        options: BayesianOptions
            Specific options for this generator
        """
        options = options or BayesianOptions()
        if not isinstance(options, BayesianOptions):
            raise ValueError("options must be a `BayesianOptions` object")

        if vocs.n_objectives != 1:
            raise ValueError("vocs must have one objective for optimization")

        super().__init__(vocs, options)
        self.algo = algo_class(**algo_kwargs)
        self.meas_param = meas_param

    @staticmethod
    def default_options() -> BayesianOptions:
        return BayesianOptions()

    def generate(self, n_candidates: int) -> List[Dict]:
        if n_candidates > 1:
            raise NotImplementedError(
                "Bayesian algorithms don't currently support parallel candidate "
                "generation"
            )

        # if no data exists use random generator to generate candidates
        if self.data.empty:
            return self.vocs.random_inputs(self.options.n_initial)

        else:
            bounds = self._get_bounds()

            start = time.time()

            # update internal model with internal data
            self.train_model(self.data)

            acq_funct = self.get_acquisition(self._model)

            middle = time.time()
            logger.debug("get_acquisition() took", middle - start, "seconds.")

            batch_initial_points, raw_samples = self._get_initial_batch_points(bounds)

            # get candidates in real domain
            candidates, out = optimize_acqf(
                acq_function=acq_funct,
                bounds=bounds,
                q=n_candidates,
                batch_initial_conditions=batch_initial_points,
                raw_samples=raw_samples,
                num_restarts=self.options.optim.num_restarts,
            )

            end = time.time()
            logger.debug("optimize_acqf() took", end - middle, "seconds.")

            logger.debug("Best candidate from optimize", candidates, out)
            return self.vocs.convert_numpy_to_inputs(candidates.detach().cpu().numpy())

    def _get_objective(self):
        return create_mc_objective(self.vocs)

    def _get_acquisition(self, model):
        SingleTaskModel = model.models[0]
        EIG = ExpectedInformationGain(SingleTaskModel, self.algo)
        return EIG

    def _get_initial_batch_points(self, bounds):
        # if self.options.optim.use_nearby_initial_points:
        if True:
            # generate starting points for optimization (note in real domain)
            inputs = self.get_input_data(self.data)
            batch_initial_points = sample_truncated_normal_perturbations(
                inputs[-1].unsqueeze(0),
                n_discrete_points=self.options.optim.raw_samples,
                sigma=0.5,
                bounds=bounds,
            ).unsqueeze(-2)
            raw_samples = None
        else:
            batch_initial_points = None
            raw_samples = self.options.optim.raw_samples

        return batch_initial_points, raw_samples
