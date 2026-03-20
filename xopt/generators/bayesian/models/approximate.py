import warnings
from copy import deepcopy
from typing import Dict, List, Union

import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Standardize
from pydantic import Field
from gpytorch.mlls import VariationalELBO

from xopt.generators.bayesian.models.standard import (
    StandardModelConstructor,
)
from xopt.generators.bayesian.utils import get_training_data


class ApproximateModelConstructor(StandardModelConstructor):
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

    name: str = Field("approximate", frozen=True)

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
                outcome_name, input_names, input_bounds, tkwargs
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

            # train basic approximate single-task-gp model
            models.append(
                self.build_variational_gp(
                    train_X.to(**tkwargs),
                    train_Y.to(**tkwargs),
                    likelihood=self.get_likelihood(),
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
            trained_models = []
            models = (
                full_model.models
                if isinstance(full_model, ModelListGP)
                else [full_model]
            )
            for m in models:
                num_data = train_X.shape[0]
                mll = VariationalELBO(m.likelihood, m.model, num_data=num_data)
                trained_models.append(self._train_model(m, mll))
            full_model = ModelListGP(*trained_models)

        # cache model hyperparameters
        self._hyperparameter_store = full_model.state_dict()

        return full_model.to(**tkwargs)
