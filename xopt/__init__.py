from xopt import _version
from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generator import Generator
from xopt.vocs import VOCS

__all__ = ["Xopt", "VOCS", "Generator", "Evaluator"]

__version__ = _version.get_versions()["version"]

from xopt.log import configure_logger


def output_notebook():
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger()
