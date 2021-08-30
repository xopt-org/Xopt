from abc import ABC, abstractmethod
from botorch.sampling.samplers import SobolQMCNormalSampler
from xopt.bayesian.utils import get_bounds
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
import torch
import logging

from typing import Dict, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class Generator(ABC):
    def __init__(self, vocs: [Dict]) -> None:
        self.vocs = vocs

    @abstractmethod
    def generate(self, model: [Model]) -> torch.Tensor:
        pass


class BayesianGenerator(Generator):
    def __init__(self,
                 vocs: [Dict],
                 acq_func: Union[Callable, AcquisitionFunction],
                 batch_size: Optional[int] = 1,
                 num_restarts: Optional[int] = 20,
                 raw_samples: Optional[int] = 1024,
                 mc_samples: Optional[int] = 512,
                 use_gpu: Optional[bool] = False,
                 acq_options: Optional[Dict] = None,
                 optimize_options: Optional[Dict] = None,
                 ) -> None:
        """

        Parameters
        ----------
        vocs : dict
            Varabiles, objectives, constraints and statics dictionary,
            see xopt documentation for detials

        acq_func : callable, AcquisitionFunction
            Botorch Acquisition function object or function that returns
            AcqusititionFunction object

        batch_size : int, default: 1
            Batch size for parallel candidate generation.

        num_restarts : int, default: 20
            Number of optimization restarts used when performing optimization(s)

        raw_samples : int, default: 1024
            Number of raw samples to use when performing optimization(s)

        mc_samples : int, default: 512
            Number of Monte Carlo samples to use during MC calculation, (ignored for
            analytical calculations)

        use_gpu : bool, default: False
            Flag to use GPU when available

        acq_options : dict, optional
            Dictionary of arguments to pass to the acquisition function

        optimize_options : dict, optional
            Extra arguments passed to optimizer(s)
        """

        super(BayesianGenerator, self).__init__(vocs)

        # check to make sure acq_function is correct type
        assert isinstance(acq_func, AcquisitionFunction) or callable(acq_func), \
            "`acq_function` must be of type AcquisitionFunction or callable"
        self.acq_func = acq_func
        self.sampler = SobolQMCNormalSampler(mc_samples)

        # configure data_type
        self.tkwargs = {"dtype": torch.double,
                        "device": torch.device("cpu")}

        # set up gpu if requested
        if use_gpu:
            if torch.cuda.is_available():
                self.tkwargs['device'] = torch.device('cuda')
                logger.info(
                    f'using gpu device '
                    f'{torch.cuda.get_device_name(self.tkwargs["device"])}')
            else:
                logger.warning('gpu requested but not found, using cpu')

        if acq_options is None:
            acq_options = {}
        if optimize_options is None:
            optimize_options = {}

        self.acq_options = acq_options

        self.optimize_options = {'q': batch_size,
                                 'num_restarts': num_restarts,
                                 'raw_samples': raw_samples,
                                 'options': {"batch_limit": 5, "maxiter": 200}}

        self.optimize_options.update(optimize_options)

    def generate(self,
                 model: [Model]) -> torch.Tensor:
        """

        Parameters
        ----------
        model : botorch.

        Returns
        -------
        candidates : torch.Tensor
            Candidates for observation
        """
        bounds = get_bounds(self.vocs, **self.tkwargs)

        # set up acquisition function object
        acq_func = self.acq_func(model, **self.acq_options)
        assert isinstance(acq_func, AcquisitionFunction), "`acq_func` method must " \
                                                          "return a botorch " \
                                                          "acquisition function type"

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            **self.optimize_options
        )

        return candidates.detach()
