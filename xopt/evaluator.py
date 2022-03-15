from concurrent.futures import Executor, ThreadPoolExecutor, Future
from threading import Lock
from typing import Callable, List, Dict


class Evaluator:
    def __init__(self, function: Callable, executor: Executor = None, max_workers=1):
        """
        light wrapper around the executor class, by default it uses a thread pool
        executor with max_workers=1

        """
        if executor is None:
            self._executor = DummyExecutor()
            self.max_workers = 1
        else:
            self._executor = executor
            self.max_workers = max_workers
        self.function = function

    def submit(self, candidates: List[Dict]):
        """submit candidates to executor"""
        futures = []
        for candidate in candidates:
            futures += [self._executor.submit(self.function, candidate)]

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
