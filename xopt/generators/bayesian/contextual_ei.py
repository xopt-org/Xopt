from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from botorch.acquisition import FixedFeatureAcquisitionFunction


class ContextualExpectedImprovementGenerator(ExpectedImprovementGenerator):
    name = "contextual_expected_improvement"

    contextual_observerables: list[str]

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function.
        Overwrites the expected improvement `get_acquisition` method.

        Parameters:
        -----------
        model : Model
            The model used for Bayesian Optimization.

        Returns:
        --------
        acq : AcquisitionFunction
            The acquisition function.
        """

        if model is None:
            raise ValueError("model cannot be None")
        
        # check to make sure that contextual observerables are in vocs
        if not set(self.contextual_observerables).issubset(set(self.vocs.observable_names)):
            raise ValueError(
                "contextual observerables must be a subset of vocs observable names"
            )

        # get the base EI acquisition function with constraints
        acq = self._get_acquisition(model)

        # get the last contextual observerable values
        last_contextual_observable_values = (
            self.data[self.contextual_observerables].iloc[-1].values
        )

        # update fixed features with the last contextual observerable values
        if self.fixed_features is None:
            self.fixed_features = {}

        self.fixed_features.update(
            {
                name: value
                for name, value in zip(
                    self.contextual_observerables, last_contextual_observable_values
                )
            }
        )

        # apply fixed features 
        dim = len(self.model_input_names)
        columns = []
        values = []
        for name, value in self.fixed_features.items():
            columns += [self.model_input_names.index(name)]
            values += [value]

        acq = FixedFeatureAcquisitionFunction(
            acq_function=acq, d=dim, columns=columns, values=values
        )

        acq = acq.to(**self.tkwargs)
        return acq
