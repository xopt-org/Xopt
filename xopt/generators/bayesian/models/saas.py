import os.path
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

from botorch.exceptions import ModelFittingError
import botorch.settings
import pandas as pd
import torch
from botorch import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.priors import GammaPrior, Prior
from pydantic import ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch.nn import Module
from torch.optim import Adam

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.generators.bayesian.utils import get_training_data, get_training_data_batched
from xopt.pydantic import XoptBaseModel, decode_torch_module

DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}
MIN_INFERRED_NOISE_LEVEL = 1e-4


class SaasModelConstructor(ModelConstructor):
    """
    A class for constructing Sparse Axis-Aligned Subspace (SAAS) models.

    Attributes
    ----------
    name : str
        The name of the model (frozen).

    use_low_noise_prior : bool
        Specify if the model should assume a low noise environment.

    trainable_mean_keys : List[str]
        List of prior mean modules that can be trained.

    transform_inputs : Union[Dict[str, bool], bool]
        Specify if inputs should be transformed inside the GP model. Can optionally
        specify a dict of specifications.

    custom_noise_prior : Optional[Prior]
        Specify a custom noise prior for the GP likelihood. Overwrites value specified
        by use_low_noise_prior.

    use_cached_hyperparameters : Optional[bool]
        Flag to specify if cached hyperparameters should be used in model creation.
        Training will still occur unless train_model is False.

    train_method : Literal["lbfgs", "adam"]
        Numerical optimization algorithm to use.

    train_model : bool
        Flag to specify if the model should be trained (fitted to data).

    train_config : NumericalOptimizerConfig
        Configuration of the numerical optimizer.

    """

    name: str = Field("saas", frozen=True)
    use_low_noise_prior: bool = Field(
        False, description="specify if model should assume a low noise environment"
    )
    trainable_mean_keys: List[str] = Field(
        [], description="list of prior mean modules that can be trained"
    )
    transform_inputs: Union[Dict[str, bool], bool] = Field(
        True,
        description="specify if inputs should be transformed inside the gp "
        "model, can optionally specify a dict of specifications",
    )
    custom_noise_prior: Optional[Prior] = Field(
        None,
        description="specify custom noise prior for the GP likelihood, "
        "overwrites value specified by use_low_noise_prior",
    )
    use_cached_hyperparameters: Optional[bool] = Field(
        False,
        description="flag to specify if cached hyperparameters should be used in "
        "model creation. Training will still occur unless train_model is False.",
    )
    train_model: bool = Field(
        True,
        description="flag to specify if the model should be trained (fitted to data)",
    )
    warmup_steps: int = Field(
        512, description="number of warmup steps to use if training with MCMC"
    )
    num_samples: int = Field(
        256, description="number of samples to use if training with MCMC"
    )
    thinning: int = Field(
        16,
        description="thinning factor to use if training with MCMC, only every nth sample is kept",
    )

    _hyperparameter_store: Optional[Dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @field_validator("trainable_mean_keys")
    def validate_trainable_mean_keys(cls, value: Any, info: ValidationInfo):
        return value

    def get_likelihood(
        self,
        batch_shape: torch.Size = torch.Size(),
    ) -> Likelihood:
        """
        Get the likelihood for the model, considering the low noise prior and or a
        custom noise prior.

        Returns
        -------
        Likelihood
            The likelihood for the model.

        """
        if self.custom_noise_prior is not None:
            likelihood = GaussianLikelihood(
                noise_prior=self.custom_noise_prior, batch_shape=batch_shape
            )
        elif self.use_low_noise_prior:
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(1.0, 100.0), batch_shape=batch_shape
            )
        else:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
                batch_shape=batch_shape,
            )
        return likelihood

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelListGP:
        """
        Construct independent models for each objective and constraint.

        Parameters
        ----------
        input_names : List[str]
            Names of input variables.
        outcome_names : List[str]
            Names of outcome variables.
        data : pd.DataFrame
            Data used for training the model.
        input_bounds : Dict[str, List], optional
            Bounds for input variables.
        dtype : torch.dtype, optional
            Data type for the model (default is torch.double).
        device : Union[torch.device, str], optional
            Device on which to perform computations (default is "cpu").

        Returns
        -------
        ModelListGP
            A list of trained botorch models.

        """
        # build model
        tkwargs = {"dtype": dtype, "device": device}
        models = []

        # validate if model caching can be used if requested
        if self.use_cached_hyperparameters:
            if self._hyperparameter_store is None:
                raise RuntimeWarning(
                    "cannot use cached hyperparameters, hyperparameter store empty, "
                    "training GP model hyperparameters instead"
                )

        for outcome_name in outcome_names:
            input_transform = self._get_input_transform(
                outcome_name, input_names, input_bounds, tkwargs
            )
            outcome_transform = Standardize(1)

            # get training data
            train_X, train_Y, train_Yvar = get_training_data(
                input_names, outcome_name, data
            )
            # collect arguments into a single dict
            kwargs = {
                "input_transform": input_transform,
                "outcome_transform": outcome_transform,
            }

            # train SAAS single-task-gp
            models.append(
                self.build_saas_gp(
                    train_X.to(**tkwargs),
                    train_Y.to(**tkwargs),
                    train_Yvar.to(**tkwargs) if train_Yvar is not None else None,
                    train=False,
                    **kwargs,
                )
            )

        full_model = ModelListGP(*models)

        # if specified, use cached model hyperparameters
        if self.use_cached_hyperparameters and self._hyperparameter_store is not None:
            store = {
                name: ele.to(**tkwargs)
                for name, ele in self._hyperparameter_store.items()
            }
            full_model.load_state_dict(store)

        if self.train_model:
            full_model = self._train_model(full_model)

        # cache model hyperparameters
        self._hyperparameter_store = full_model.state_dict()

        return full_model.to(**tkwargs)

    def _train_model(self, model):
        models = model.models if isinstance(model, ModelListGP) else [model]

        for m in models:
            try:
                fit_fully_bayesian_model_nuts(
                    m,
                    warmup_steps=self.warmup_steps,
                    num_samples=self.num_samples,
                    thinning=self.thinning,
                    disable_progbar=True,
                )
            except ModelFittingError:
                warnings.warn("Model fitting failed. Returning untrained model.")
        return model

    @staticmethod
    def _get_module(base, name):
        """
        Get the module for a given name.

        Parameters
        ----------
        base : Union[Module, Dict[str, Module]]
            The base module or a dictionary of modules.
        name : str
            The name of the module.

        Returns
        -------
        Module
            The retrieved module.

        """
        if isinstance(base, Module):
            return deepcopy(base)
        elif isinstance(base, dict):
            return deepcopy(base.pop(name, None))
        else:
            return None

    def _get_input_transform(self, outcome_name, input_names, input_bounds, tkwargs):
        """
        Get input transform based on the supplied bounds and attributes

        Parameters
        ----------
        outcome_name : str
            The name of the outcome variable.
        input_names : list[str]
            The names of the input variables.
        input_bounds : dict[str, tuple[float, float]]
            The bounds for the input variables.
        tkwargs : dict
            Additional keyword arguments for tensor creation.

        """
        # get input bounds
        if input_bounds is None:
            bounds = None
        else:
            bounds = torch.vstack(
                [torch.tensor(input_bounds[name], **tkwargs) for name in input_names]
            ).T

        # create transform
        input_transform = Normalize(len(input_names), bounds=bounds)

        # remove input transform if the bool is False or the dict entry is false
        if isinstance(self.transform_inputs, bool):
            if not self.transform_inputs:
                input_transform = None
        if (
            isinstance(self.transform_inputs, dict)
            and outcome_name in self.transform_inputs
        ):
            if not self.transform_inputs[outcome_name]:
                input_transform = None

        # remove warnings if input transform is None
        if input_transform is None:
            botorch.settings.validate_input_scaling(False)

        return input_transform


class BatchedSaasModelConstructor(SaasModelConstructor):
    """
    BatchedModelConstructor treats outputs as an additional dimension instead of looping over them.
    It is useful when multiple outputs are being modelled and their settings are similar.

    A batch shares training points (i.e. train_X/train_Y) and kernel. It uses a single pytorch
    graph (1 sum loss), which changes convergence criteria. Resulting GP parameters are expected
    to differ slightly from individual fitting.

    Batch modelling is faster on GPUs, especially in the intermediate (<~1000) problem sizes where GPU
    call overhead is significant and all models converge in roughly equal number of steps.

    On CPU, batched fitting performance varies from slightly useful to slightly detrimental - mostly
    not worth it.

    See benchmarking docs on how to run tests for your specific hardware.
    """

    def _get_input_transform(
        self,
        outcome_names: list[str],
        input_names: list[str],
        input_bounds,
        batch_shape: torch.Size = torch.Size(),
    ) -> Optional[Normalize]:
        if input_bounds is None:
            bounds = None
        else:
            bounds = torch.vstack(
                [torch.tensor(input_bounds[name]) for name in input_names]
            ).T

        input_transform = Normalize(
            len(input_names),
            bounds=bounds,
            batch_shape=batch_shape,
        )

        # remove input transform if the bool is False or the dict entry is false
        if isinstance(self.transform_inputs, bool):
            if not self.transform_inputs:
                input_transform = None

        if isinstance(self.transform_inputs, dict):
            raise AttributeError(
                "Cannot specify dict for transform_inputs when using BatchedModelConstructor"
            )

        if input_transform is None:
            botorch.settings.validate_input_scaling(False)

        return input_transform

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> SingleTaskGP:
        """
        Construct a single batched model for all objectives and constraints.
        """
        tkwargs = {"dtype": dtype, "device": device}

        if self.use_cached_hyperparameters:
            if self._hyperparameter_store is None:
                raise RuntimeWarning(
                    "cannot use cached hyperparameters, hyperparameter store empty, "
                    "training GP model hyperparameters instead"
                )

        train_X, train_Y, train_Yvar = get_training_data_batched(
            input_names, outcome_names, data
        )
        if train_X.shape[0] == 0 or train_Y.shape[0] == 0:
            raise ValueError("no data found to train model!")

        # train_Y is n x m, will get transformed to (m) x n x 1 by
        # SingleTaskGP to run as m independent batches
        # _input_batch_shape = empty
        # _aug_batch_shape = (m,) = (_num_outputs,)

        _num_outputs = train_Y.shape[-1]
        _input_batch_shape, _aug_batch_shape = (
            BatchedMultiOutputGPyTorchModel.get_batch_dimensions(
                train_X=train_X, train_Y=train_Y
            )
        )
        # input and output transforms are applied BEFORE tensors are unrolled
        input_transform = self._get_input_transform(
            outcome_names,
            input_names,
            input_bounds=input_bounds,
            batch_shape=_input_batch_shape,
        )
        outcome_transform = Standardize(
            m=train_Y.shape[-1], batch_shape=_input_batch_shape
        )
        kwargs = {
            "input_transform": input_transform,
            "outcome_transform": outcome_transform,
        }

        if train_Yvar is None:
            likelihood = self.get_likelihood(batch_shape=_aug_batch_shape)
        else:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=train_Yvar, batch_shape=_aug_batch_shape
            )
        full_model = SingleTaskGP(
            train_X, train_Y, train_Yvar=train_Yvar, likelihood=likelihood, **kwargs
        )
        full_model.to(**tkwargs)

        if self.use_cached_hyperparameters and self._hyperparameter_store is not None:
            store = {
                name: ele.to(**tkwargs)
                for name, ele in self._hyperparameter_store.items()
            }
            full_model.load_state_dict(store)

        if self.train_model:
            full_model = self._train_model(full_model)

        # cache model hyperparameters
        self._hyperparameter_store = full_model.state_dict()

        return full_model
