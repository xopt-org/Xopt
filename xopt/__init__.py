from . import _version
__version__ = _version.get_versions()['version']

from .xopt import Xopt
from .tools import xopt_logo


from xopt.log import configure_logger

def output_notebook():
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger()
    
    
