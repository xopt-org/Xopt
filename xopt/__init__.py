from .evaluator import Evaluator
from .generator import Generator
from .vocs import VOCS
from .base import XoptBase

from . import _version
__version__ = _version.get_versions()['version']

from xopt.log import configure_logger

def output_notebook():
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger()
