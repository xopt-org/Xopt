import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.means import Mean

from xopt.generators.bayesian.models.transformed_model import TransformedModel


class CustomMean(TransformedModel, Mean):
    def __init__(
        self,
        model: torch.nn.Module,
        gp_input_transform: InputTransform,
        gp_outcome_transform: OutcomeTransform,
        training_intended: bool = False
    ):
        """Custom prior mean for a GP based on an arbitrary model.

        Args:
            model: Representation of the model.
            gp_input_transform: Module used to transform inputs in the GP.
            gp_outcome_transform: Module used to transform outcomes in the GP.
            training_intended: Whether training the model is intended. If False,
              the model is put in evaluation mode and gradient computation is
              deactivated.
        """
        super().__init__(model, gp_input_transform, gp_outcome_transform,
                         training_intended=training_intended)

    def forward(self, x):
        # set transformers to eval mode
        self.set_transformers(eval_mode=True)

        # transform inputs to model space
        x_model = self.input_transformer.untransform(x)

        # evaluate model
        y_model = self.evaluate_model(x_model)

        # transform outputs
        y = self.outcome_transformer(y_model)[0].squeeze(dim=0)

        self.set_transformers(eval_mode=False)

        return y
