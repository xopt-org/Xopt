import logging
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from enum import Enum
from threading import Lock
from typing import Callable, Dict

import pandas as pd
from pydantic import BaseModel, Field, root_validator

from xopt.pydantic import JSON_ENCODERS, NormalExecutor
from xopt.utils import get_function, get_function_defaults

logger = logging.getLogger(__name__)


class ExecutorEnum(str, Enum):
    thread_pool_executor = "ThreadPoolExecutor"
    process_pool_executor = "ProcessPoolExecutor"
    normal_executor = "NormalExecutor"


class Evaluator(BaseModel):
    """
    Xopt Evaluator for handling the parallel execution of an evaluate function.

    Parameters
    ----------
    function : Callable
        Function to evaluate.
    function_kwargs : dict, default={}
        Any kwargs to pass on to this function.
    max_workers : int, default=1
        Maximum number of workers.
    executor : NormalExecutor
        NormalExecutor or any instantiated Executor object
    """

    function: Callable
    max_workers: int = 1
    executor: NormalExecutor = Field(exclude=True)  # Do not serialize
    function_kwargs: dict = {}

    class Config:
        """config"""
        arbitrary_types_allowed = True
        # validate_assignment = True # Broken in 1.9.0.
        # Trying to fix in https://github.com/samuelcolvin/pydantic/pull/4194
        json_encoders = JSON_ENCODERS
        extra = "forbid"
        # copy_on_model_validation = False

    @root_validator(pre=True)
    def validate_all(cls, values):

        f = get_function(values["function"])
        kwargs = values.get("function_kwargs", {})
        kwargs = {**get_function_defaults(f), **kwargs}
        values["function"] = f
        values["function_kwargs"] = kwargs

        max_workers = values.pop("max_workers", 1)

        executor = values.pop("executor", None)
        if not executor:
            if max_workers > 1:
                executor = ProcessPoolExecutor(max_workers=max_workers)
            else:
                executor = DummyExecutor()

        # Cast as a NormalExecutor
        values["executor"] = NormalExecutor[type(executor)](executor=executor)
        values["max_workers"] = max_workers

        return values

    def evaluate(self, input: Dict, **kwargs):
        """
        Evaluate a single input dict using Evaluator.function with
        Evaluator.function_kwargs.

        Further kwargs are passed to the function.

        Inputs:
            inputs: dict of inputs to be evaluated
            **kwargs: additional kwargs to pass to the function

        Returns:
            function(input, **function_kwargs_updated)

        """
        return self.function(input, **{**self.function_kwargs, **kwargs})

    def submit(self, input: Dict):
        """submit a single input to the executor"""
        if not isinstance(input, dict):
            raise ValueError("input must be a dictionary")
        return self.executor.submit(self.function, input, **self.function_kwargs)

    def submit_data(self, input_data: pd.DataFrame):
        """submit dataframe of inputs to executor"""
        input_data = pd.DataFrame(input_data)  # cast to dataframe
        futures = {}
        for index, row in input_data.iterrows():
            future = self.submit(dict(row))
            futures[index] = future

        return futures


class DummyExecutor(Executor):
    """
    Dummy executor.

    From: https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures

    """

    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
