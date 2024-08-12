from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from xopt.generators.bayesian.bayesian_generator import MultiObjectiveBayesianGenerator

from xopt.generators.bayesian.objectives import create_mobo_objective


class MOBOGenerator(MultiObjectiveBayesianGenerator):
    name = "mobo"
    __doc__ = """Implements Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""

    def _get_objective(self):
        return create_mobo_objective(self.vocs)

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply fixed features if specified in the generator
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns += [self.model_input_names.index(name)]
                values += [value]

            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )

        return acq

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        if self.log_transform_acquisition_function:
            acqclass = qLogNoisyExpectedHypervolumeImprovement
        else:
            acqclass = qNoisyExpectedHypervolumeImprovement

        acq = acqclass(
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
