from xopt import _version
from xopt.base import Xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generator import Generator

__version__ = _version.get_versions()['version']

from xopt.log import configure_logger

def output_notebook():
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger()
