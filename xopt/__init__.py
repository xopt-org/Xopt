from xopt import _version
from xopt.asynchronous import AsynchronousXopt
from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generator import Generator
from xopt.vocs import VOCS

__all__ = ["Xopt", "VOCS", "Generator", "Evaluator", "AsynchronousXopt"]

__version__ = _version.get_versions()["version"]

from xopt.log import configure_logger


def output_notebook(**kwargs):
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger(**kwargs)


def from_file(file_path, asynchronous=False):
    if asynchronous:
        return AsynchronousXopt.from_file(file_path)
    else:
        return Xopt.from_file(file_path)
