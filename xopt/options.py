from enum import Enum

from pydantic import BaseModel, Field


class LoggingEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class XoptOptions(BaseModel):
    asynch: bool = Field(
        False, description="flag to evaluate and submit evaluations asynchronously"
    )
    strict: bool = Field(
        False,
        description="flag to indicate if exceptions raised during evaluation "
        "should stop Xopt",
    )
    timeout: float = Field(
        None, description="maximum waiting time during `Xopt.step()`"
    )
    dump_file: str = Field(
        None, description="file to dump the results of the evaluations"
    )
    #logging_level: LoggingEnum = Field(
    #    LoggingEnum.INFO, description="logging level for package logging"
    #)
    #logging_file: str = Field(None, description="file to dump the logs of the package")

    #class Config:
    #    use_enum_values = True
