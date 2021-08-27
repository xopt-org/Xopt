from abc import ABC, abstractmethod
from botorch.sampling.samplers import SobolQMCNormalSampler
from xopt.bayesian.utils import get_bounds
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import AcquisitionFunction
import torch
import logging

logger = logging.getLogger(__name__)


class Generator(ABC):
    def __init__(self, vocs):
        self.vocs = vocs

    @abstractmethod
    def generate(self):
        pass


class BayesianGenerator(Generator):
    def __init__(self, vocs,
                 acq_func,
                 batch_size=1,
                 num_restarts=20,
                 raw_samples=1024,
                 mc_samples=512,
                 use_gpu=False,
                 acq_options=None,
                 optimize_options=None,
                 ):
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

    def generate(self, model):
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
