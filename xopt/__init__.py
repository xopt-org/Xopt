from .xopt import Xopt
from .tools import xopt_logo
from ._version import __version__

from . import _version
__version__ = _version.get_versions()['version']
