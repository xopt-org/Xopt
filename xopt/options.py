from pydantic import BaseModel, Field


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
