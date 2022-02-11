from concurrent.futures import Executor
from typing import Callable, List, Dict


class Evaluator:
    def __init__(self, function: Callable, executor: Executor):
        """
        light wrapper around the executor class which checks if the function passed
        to it has the right form


        """
        self._executor = executor
        self.function = function

    def submit(self, candidates: List[Dict]):
        """submit candidates to executor"""
        futures = []
        for candidate in candidates:
            futures += [self._executor.submit(self.function, **candidate)]

        return futures
