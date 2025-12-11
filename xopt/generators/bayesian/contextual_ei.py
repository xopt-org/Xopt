from copy import deepcopy
from pandas import DataFrame
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from botorch.acquisition import FixedFeatureAcquisitionFunction


class ContextualExpectedImprovementGenerator(UpperConfidenceBoundGenerator):
    name = "contextual_expected_improvement"

    contextual_observables: list[str]

    @property
    def model_input_names(self):
        """
        Returns the names of the model inputs, which include both variables and contextual observables.
        """
        input_names = super().model_input_names
        return input_names + self.contextual_observables

    def get_model_input_bounds(self, data: DataFrame) -> dict[str, tuple[float, float]]:
        """
        Returns the bounds for the model inputs, including contextual observables.
        Context variable bounds are determined from the data provided.
        """
        bounds = super().get_model_input_bounds(data)
        for obs in self.contextual_observables:
            bounds[obs] = (
                data[obs].min(),
                data[obs].max() + 1e-6,
            )  # small epsilon to avoid zero range
        return bounds

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

        # check to make sure that contextual observables are in vocs
        if not set(self.contextual_observables).issubset(
            set(self.vocs.observable_names)
        ):
            raise ValueError(
                "contextual observables must be a subset of vocs observable names"
            )

        # get the base EI acquisition function with constraints
        acq = self._get_acquisition(model)

        # get the last contextual observable values
        last_contextual_observable_values = (
            self.data[self.contextual_observables].iloc[-1].values
        )

        # update fixed features with the last contextual observable values
        if self.fixed_features is None:
            self.fixed_features: dict[str, float] = {}

        fixed_features = deepcopy(self.fixed_features)

        fixed_features.update(
            {
                name: value
                for name, value in zip(
                    self.contextual_observables, last_contextual_observable_values
                )
            }
        )

        # apply fixed features
        dim = len(self.model_input_names)
        columns = []
        values = []
        for name, value in fixed_features.items():
            columns += [self.model_input_names.index(name)]
            values += [value]

        acq = FixedFeatureAcquisitionFunction(
            acq_function=acq, d=dim, columns=columns, values=values
        )

        acq = acq.to(**self.tkwargs)
        return acq
