from gpytorch.priors import Prior
from torch.distributed.pipeline.sync.batchnorm import TModule
from torch.distributions import HalfCauchy


class HalfCauchyPrior(Prior, HalfCauchy):
    """
    Half-Cauchy prior.
    """

    def __init__(self, scale, validate_args=None, transform=None):
        TModule.__init__()
        HalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape, **kwargs):
        return HalfCauchy(self.loc.expand(batch_shape), self.scale.expand(batch_shape))
