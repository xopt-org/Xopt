import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.means import Mean


class CustomMean(Mean):
    def __init__(
        self,
        model: torch.nn.Module,
        input_transformer: InputTransform,
        outcome_transformer: OutcomeTransform,
        fixed_model: bool = True,
    ):
        """Custom prior mean for a GP based on an arbitrary model.

        Parameters
        ----------
            model: Representation of the model.
            input_transformer: Module used to transform inputs in the GP.
            outcome_transformer: Module used to transform outcomes in the GP.
            fixed_model: If true, the model is put in evaluation mode and
              gradient computation is deactivated. Note that, even if this is
              set to false, model inference will always happen in evaluation
              mode unless training mode is reactivated in the base model.
        """
        super().__init__()
        self._model = model
        if fixed_model:
            self._model.eval()
            self._model.requires_grad_(False)
        self.input_transformer = input_transformer
        self.outcome_transformer = outcome_transformer

    @property
    def model(self):
        return self._model

    def get_transformer_states(self) -> list[dict, dict]:
        states = [None, None]
        transformers = [self.input_transformer, self.outcome_transformer]
        for i, t in enumerate(transformers):
            if isinstance(t, torch.nn.Module):
                g = {}
                for name, param in t.named_parameters():
                    g[name] = param.requires_grad
                states[i] = {
                    "training": t.training,
                    "gradients": g,
                }
        return states

    def set_transformer_states(self, states: list[dict, dict]):
        transformers = [self.input_transformer, self.outcome_transformer]
        for i, t in enumerate(transformers):
            if isinstance(t, torch.nn.Module):
                for name, param in t.named_parameters():
                    param.requires_grad = states[i]["gradients"][name]
                t.train(mode=states[i]["training"])

    def deactivate_transformer_gradients(self):
        transformers = [self.input_transformer, self.outcome_transformer]
        for i, t in enumerate(transformers):
            if isinstance(t, torch.nn.Module):
                t.requires_grad_(False)

    def forward(self, x):
        # get initial transformer states
        transformer_states = self.get_transformer_states()

        # set model and transformers in eval mode
        # otherwise GP training will activate layers like Dropout, BatchNorm etc.
        # if this behavior is intended, training mode should be reactivated in the base model
        self.eval()

        # deactivate transformer gradients
        self.deactivate_transformer_gradients()

        # transform inputs to model space
        x_model = self.input_transformer.untransform(x)

        # evaluate model
        y_model = self._model(x_model)

        # transform outputs
        y = self.outcome_transformer(y_model)[0].squeeze(dim=0)

        # set transformers back to initial state
        self.set_transformer_states(transformer_states)

        return y
