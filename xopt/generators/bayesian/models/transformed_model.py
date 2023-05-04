import torch


class TransformedModel(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            input_transformer,
            outcome_transformer,
            fixed_model: bool = True
    ):
        """A model that requires an input and outcome transform to evaluate.

        Transformer objects must have a forward method that transforms tensors
        into input coordinates for model and an untransform method that does
        the reverse.

        model: Representation of the model.
        input_transformer: Module used to transform inputs.
        outcome_transformer: Module used to transform outcomes.
        fixed_model: If true, the model is put in evaluation mode and
          gradient computation is deactivated.
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

    def set_transformers(self, eval_mode=True):
        """set transformers to eval mode if they are torch.nn.Module objects -
        prevents transformer training during evaluation
        if `eval_mode` is true set to eval, otherwise set to train
        """
        modules = [self.input_transformer, self.outcome_transformer]
        for m in modules:
            if isinstance(m, torch.nn.Module):
                if eval_mode:
                    m.eval()
                else:
                    m.train()

    def evaluate_model(self, x):
        """Placeholder method which can be used to modify model calls."""
        return self._model(x)

    def forward(self, x):
        # set transformers to eval mode
        self.set_transformers(eval_mode=True)

        # transform inputs to model space
        x_model = self.input_transformer(x)

        # evaluate model
        y_model = self.evaluate_model(x_model)

        # transform outputs
        y = self.outcome_transformer.untransform(y_model)

        self.set_transformers(eval_mode=False)

        return y
