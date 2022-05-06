from concurrent.futures import Executor, ThreadPoolExecutor, Future
from threading import Lock
from typing import Callable, List, Dict

import pandas as pd
from xopt.pydantic import XoptBaseModel


class EvaluatorOptions(XoptBaseModel):
    """
    Evaluator model.
    """
    function: Callable = None
    max_workers: int = 1


class Evaluator:
    def __init__(self, function: Callable, executor: Executor = None, max_workers=1):
        """
        light wrapper around the executor class, by default it uses a dummy 
        executor with max_workers=1

        """
        if executor is None:
            self._executor = DummyExecutor()
            self.max_workers = 1
        else:
            self._executor = executor
            self.max_workers = max_workers
        self.function = function

        self._n_submitted = 0

    def submit(self, input: Dict):
        """submit a single input to the executor"""
        if not isinstance(input, dict):
            raise ValueError("input must be a dictionary")
        return self._executor.submit(self.function, input)

    def submit_data(self, input_data: pd.DataFrame):
        """submit dataframe of inputs to executor"""
        input_data = pd.DataFrame(input_data) # cast to dataframe
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
