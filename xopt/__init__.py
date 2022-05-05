from xopt import _version

__version__ = _version.get_versions()['version']

from xopt.log import configure_logger


def output_notebook():
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger()


# globally modify pydantic base model to not allow extra keys
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        extra = 'forbid'
