from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement

from xopt.generators.bayesian.objectives import create_mobo_objective
from .bayesian_generator import MultiObjectiveBayesianGenerator


class MOBOGenerator(MultiObjectiveBayesianGenerator):
    name = "mobo"
    __doc__ = """Implements Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""

    def _get_objective(self):
        return create_mobo_objective(self.vocs, self._tkwargs)

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)
        return acq

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        # fix problem with qNEHVI interpretation with constraints
        acq = qNoisyExpectedHypervolumeImprovement(
            model,
            X_baseline=inputs,
            constraints=self._get_constraint_callables(),
            ref_point=self.torch_reference_point,
            sampler=sampler,
            objective=self._get_objective(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq
