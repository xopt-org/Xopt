from abc import ABC, abstractmethod
from botorch.sampling.samplers import SobolQMCNormalSampler


class Generator(ABC):
    def __init__(self, vocs):
        self.vocs = vocs

    @abstractmethod
    def generate(self):
        pass


class BayesianGenerator(Generator, ABC):
    def __init__(self, vocs,
                 batch_size,
                 mc_samples,
                 num_restarts,
                 raw_samples,
                 **kwargs
                 ):
        super(BayesianGenerator, self).__init__(vocs)
        self.batch_size = batch_size
        self.sampler = SobolQMCNormalSampler(mc_samples)
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    @abstractmethod
    def generate(self, model, **tkwargs):
        pass
