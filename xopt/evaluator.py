from concurrent.futures import Executor, ThreadPoolExecutor, Future, ProcessPoolExecutor
from enum import Enum
from threading import Lock
from typing import Callable, List, Dict, Any
from inspect import getfile
from types import FunctionType

import pandas as pd
from pydantic import BaseModel

from xopt.pydantic import XoptBaseModel


class ExecutorEnum(str, Enum):
    thread_pool_executor = "ThreadPoolExecutor"
    process_pool_executor = "ProcessPoolExecutor"
    normal_executor = "NormalExecutor"


class EvaluatorOptions(XoptBaseModel):
    """
    Evaluator model.
    """

    function: Callable[[...], dict]
    function_kwargs: dict = {}
    max_workers: int = 1
    executor: ExecutorEnum = ExecutorEnum.normal_executor

    class Config:
        use_enum_values = True
        json_encoders = {FunctionType: lambda x: x.__module__ + "." + x.__name__}


class Evaluator:
    def __init__(
        self,
        function: Callable,
        max_workers: int = 1,
        executor: ExecutorEnum = ExecutorEnum.normal_executor,
    ):
        """
        wrapper around the executor class, by default it uses a dummy
        executor with max_workers=1

        """
        self.options = EvaluatorOptions(
            function=function, max_workers=max_workers, executor=executor
        )
        if self.options.executor == ExecutorEnum.normal_executor:
            self._executor = DummyExecutor()
            self.max_workers = 1
        elif self.options.executor == ExecutorEnum.thread_pool_executor:
            self._executor = ThreadPoolExecutor(max_workers=self.options.max_workers)
            self.max_workers = self.options.max_workers
        elif self.options.executor == ExecutorEnum.process_pool_executor:
            self._executor = ProcessPoolExecutor(max_workers=self.options.max_workers)
        self.function = self.options.function

        self._n_submitted = 0

    @classmethod
    def from_options(cls, options: EvaluatorOptions):
        return cls(
            function=options.function,
            max_workers=options.max_workers,
            executor=options.executor,
        )

    def submit(self, input: Dict):
        """submit a single input to the executor"""
        if not isinstance(input, dict):
            raise ValueError("input must be a dictionary")
        return self._executor.submit(self.function, input)

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
