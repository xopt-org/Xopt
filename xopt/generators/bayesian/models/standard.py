import os.path
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

import botorch.settings
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
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

# TODO: make custom stopping criterion that checks lengthscales


class ExpMAStoppingCriterionModel(XoptBaseModel):
    maxiter: int = Field(500, description="maximum number of iterations")
    n_window: int = Field(
        5, description="size of the exponential moving average window"
    )
    eta: float = Field(1.0, description="exponential decay factor in the weights")
    rel_tol: float = Field(5e-4, description="relative tolerance for termination")


class NumericalOptimizerConfig(XoptBaseModel):
    timeout: float | None = Field(default=None, description="timeout in seconds")


class LBFGSNumericalOptimizerConfig(NumericalOptimizerConfig):
    gtol: float = Field(
        default=1e-5, description="projected gradient tolerance, scipy default is 1e-5"
    )
    ftol: float = Field(
        default=2.2e-9,
        description="function tolerance, scipy default is 1e7 * np.finfo(float).eps = "
        "2.2204460492503131e-09",
    )
    maxiter: int = Field(default=500, description="maximum number of iterations")


class AdamNumericalOptimizerConfig(NumericalOptimizerConfig):
    stopping_criterion: ExpMAStoppingCriterionModel = Field(
        default_factory=ExpMAStoppingCriterionModel
    )
    lr: float = Field(default=0.1, description="learning rate for the Adam optimizer")


class StandardModelConstructor(ModelConstructor):
    """
    A class for constructing independent models for each objective and constraint.

    Attributes
    ----------
    name : str
        The name of the model (frozen).

    use_low_noise_prior : bool
        Specify if the model should assume a low noise environment.

    covar_modules : Dict[str, Kernel]
        Covariance modules for GP models.

    mean_modules : Dict[str, Module]
        Prior mean modules for GP models.

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

    name: str = Field("standard", frozen=True)
    use_low_noise_prior: bool = Field(
        False, description="specify if model should assume a low noise environment"
    )
    covar_modules: Dict[str, Kernel] = Field(
        {}, description="covariance modules for GP models"
    )
    mean_modules: Dict[str, Module] = Field(
        {}, description="prior mean modules for GP models"
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
    train_method: Literal["lbfgs", "adam"] = Field(
        "lbfgs", description="numerical optimization algorithm to use"
    )
    train_model: bool = Field(
        True,
        description="flag to specify if the model should be trained (fitted to data)",
    )
    train_config: NumericalOptimizerConfig | None = Field(
        None,
        description="configuration of the numerical optimizer - see fit_gpytorch_mll_scipy"
        " and fit_gpytorch_mll_torch",
    )
    train_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="additional keyword arguments passed to the training optimizer",
    )
    _hyperparameter_store: Optional[Dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @field_validator("train_kwargs")
    def validate_train_kwargs(cls, train_kwargs, info: ValidationInfo):
        if train_kwargs is None:
            return train_kwargs
        # keys are from _fit_fallback in botorch/fit.py - we don't use other dispatchers
        allowed_keys = [
            "pick_best_of_all_attempts",
            "max_attempts",
            "warning_handler",
            "optimizer_kwargs",
        ]
        allowed_subkeys = {}
        if not isinstance(train_kwargs, dict):
            raise ValueError(f"train_kwargs must be a dict, not {type(train_kwargs)}")
        invalid_keys = set(train_kwargs.keys()) - set(allowed_keys)
        if invalid_keys:
            raise ValueError(
                f"train_kwargs can only contain the keys {allowed_keys}, have {invalid_keys}"
            )
        for k, v in train_kwargs.items():
            if k in allowed_subkeys and isinstance(v, dict):
                allowed = allowed_subkeys.get(k, [])
                if set(v.keys()) - set(allowed):
                    raise ValueError(
                        f"train_kwargs['{k}'] can only contain the keys {allowed}"
                    )
        return train_kwargs

    @field_validator("train_config")
    def validate_train_config(cls, v, info: ValidationInfo):
        if v is None:
            return v
        if info.data["train_method"] == "adam":
            if not isinstance(v, AdamNumericalOptimizerConfig):
                raise ValueError(
                    "train_config must be of type AdamOptimizerConfig when method is 'adam'"
                )
        elif info.data["train_method"] == "lbfgs":
            if not isinstance(v, LBFGSNumericalOptimizerConfig):
                raise ValueError(
                    "train_config must be of type LBFGSOptimizerConfig when method is 'lbfgs'"
                )
        else:
            raise ValueError("method must be either 'adam' or 'lbfgs'")
        return v

    @field_validator("covar_modules", "mean_modules", mode="before")
    def validate_torch_modules(cls, value: Any):
        if not isinstance(value, dict):
            raise ValueError("must be dict")
        else:
            value = cast(dict[str, Any], value)
            for key, val in value.items():
                if isinstance(val, str):
                    if val.startswith("base64:"):
                        value[key] = decode_torch_module(val)
                    elif os.path.exists(val):
                        value[key] = torch.load(val, weights_only=False)

        return value

    @field_validator("trainable_mean_keys")
    def validate_trainable_mean_keys(cls, value: Any, info: ValidationInfo):
        for name in value:
            assert name in info.data["mean_modules"]
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

        covar_modules = deepcopy(self.covar_modules)
        mean_modules = deepcopy(self.mean_modules)
        for outcome_name in outcome_names:
            input_transform = self._get_input_transform(
                outcome_name, input_names, input_bounds
            )
            outcome_transform = Standardize(1)
            covar_module = self._get_module(covar_modules, outcome_name)
            mean_module = self.build_mean_module(
                outcome_name, mean_modules, input_transform, outcome_transform
            )

            # get training data
            train_X, train_Y, train_Yvar = get_training_data(
                input_names, outcome_name, data
            )
            # collect arguments into a single dict
            kwargs = {
                "input_transform": input_transform,
                "outcome_transform": outcome_transform,
                "covar_module": covar_module,
                "mean_module": mean_module,
            }

            if train_Yvar is None:
                # train basic single-task-gp model
                models.append(
                    self.build_single_task_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        likelihood=self.get_likelihood(),
                        train=False,
                        **kwargs,
                    )
                )
            else:
                # train heteroskedastic single-task-gp model
                # turn off warnings
                models.append(
                    self.build_heteroskedastic_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        train_Yvar.to(**tkwargs),
                        train=False,
                        **kwargs,
                    )
                )
        # check all specified modules were added to the model
        if covar_modules:
            warnings.warn(
                f"Covariance modules for output names {[k for k, v in self.covar_modules.items()]} "
                f"could not be added to the model."
            )
        if mean_modules:
            warnings.warn(
                f"Mean modules for output names {[k for k, v in self.mean_modules.items()]} "
                f"could not be added to the model."
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
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            tr_kwargs = self.train_kwargs if self.train_kwargs is not None else {}
            if "optimizer_kwargs" not in tr_kwargs:
                tr_kwargs["optimizer_kwargs"] = {}
            if self.train_config is not None and self.train_config.timeout is not None:
                tr_kwargs["optimizer_kwargs"]["timeout_sec"] = self.train_config.timeout
            if self.train_method == "adam":
                cfg_adam: AdamNumericalOptimizerConfig = (
                    self.train_config or AdamNumericalOptimizerConfig()
                )
                sc = ExpMAStoppingCriterion(
                    maxiter=cfg_adam.stopping_criterion.maxiter,
                    n_window=cfg_adam.stopping_criterion.n_window,
                    eta=cfg_adam.stopping_criterion.eta,
                    rel_tol=cfg_adam.stopping_criterion.rel_tol,
                )
                opt = partial(Adam, lr=cfg_adam.lr)
                tr_kwargs["optimizer_kwargs"]["stopping_criterion"] = sc
                tr_kwargs["optimizer_kwargs"]["optimizer"] = opt
                optimizer = fit_gpytorch_mll_torch
            else:
                cfg_lbfgs: LBFGSNumericalOptimizerConfig = (
                    self.train_config or LBFGSNumericalOptimizerConfig()
                )
                if "options" not in tr_kwargs["optimizer_kwargs"]:
                    tr_kwargs["optimizer_kwargs"]["options"] = {}
                tr_kwargs["optimizer_kwargs"]["options"]["maxiter"] = cfg_lbfgs.maxiter
                tr_kwargs["optimizer_kwargs"]["options"]["gtol"] = cfg_lbfgs.gtol
                tr_kwargs["optimizer_kwargs"]["options"]["ftol"] = cfg_lbfgs.ftol
                optimizer = fit_gpytorch_mll_scipy

            fit_gpytorch_mll(mll, optimizer=optimizer, **tr_kwargs)
        return model

    def build_mean_module(
        self, name, mean_modules, input_transform, outcome_transform
    ) -> Optional[CustomMean]:
        """
        Build the mean module for the output specified by name.

        Parameters
        ----------
        name : str
            The name of the output.
        mean_modules: dict
            The dictionary of mean modules.
        input_transform : InputTransform
            Transform for input variables.
        outcome_transform : OutcomeTransform
            Transform for outcome variables.

        Returns
        -------
        Optional[CustomMean]
            The mean module for the output, or None if not specified.

        """
        mean_module = self._get_module(mean_modules, name)
        if mean_module is not None:
            fixed_model = False if name in self.trainable_mean_keys else True
            mean_module = CustomMean(
                mean_module, input_transform, outcome_transform, fixed_model=fixed_model
            )
        return mean_module

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

    def _get_input_transform(self, outcome_name, input_names, input_bounds):
        """get input transform based on the supplied bounds and attributes"""
        # get input bounds
        if input_bounds is None:
            bounds = None
        else:
            bounds = torch.vstack(
                [
                    torch.tensor(input_bounds[name], dtype=torch.double)
                    for name in input_names
                ]
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


class BatchedModelConstructor(StandardModelConstructor):
    """
    BatchedModelConstructor treats outputs as an additional dimension instead of looping over them.
    It is useful when multiple outputs are being modelled and their settings are similar.

    Batch shares training points (i.e. all not-nan inputs) and kernel. It uses a single pytorch
    graph (i.e. 1 sum loss), which changes convergence criteria. Resulting parameters are expected
    to differ slightly from individual fitting.

    Batch modelling is faster on GPUs, especially in the intermediate (<~1000) problem sizes where GPU
    call overhead is significant and all models converge in roughly equal number of steps.
    On CPU, the performance gains are minimal. See benchmarking docs on how to run tests
    for your specific hardware.
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

        covar_module = None
        if len(self.covar_modules) > 1:
            raise ValueError(
                "Covariance modules cannot be specified individually when using BatchedModelConstructor"
            )
        elif len(self.covar_modules) == 1:
            covar_module = list(self.covar_modules.values())[0]

        mean_module = None
        if len(self.mean_modules) > 1:
            raise ValueError(
                "Mean modules cannot be specified individually when using BatchedModelConstructor"
            )
        # assume that if have 1 module, it is to be used for all outputs
        elif len(self.mean_modules) == 1:
            mean_module = list(self.mean_modules.values())[0]

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
            "covar_module": covar_module,
            "mean_module": mean_module,
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
