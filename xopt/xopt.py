import concurrent
import logging
from typing import Type

import pandas as pd
from .generator import Generator
from .evaluator import Evaluator
from .vocs import VOCS

logger = logging.getLogger(__name__)


class Xopt:
    """
    
    Object to handle a single optimization problem.
    
    Parameters
    ----------
    config: dict, YAML text, JSON text
        input file should be a dict, JSON, or YAML file with top level keys
    
          
    """
    _futures = []
    _samples = []
    _history = None
    _is_done = False
    timeout = 1.0

    def __init__(
            self,
            generator: Generator, evaluator: Evaluator, vocs: VOCS,
            asynch=False
    ):
        # initialize Xopt object
        self._generator = generator
        self._evaluator = evaluator
        self._vocs = vocs
        self.asynch = asynch

        if self.asynch:
            self.return_when = concurrent.futures.FIRST_COMPLETED
        else:
            self.return_when = concurrent.futures.ALL_COMPLETED

    def run(self):
        """run until either xopt is done or the generator is done"""
        while not self._is_done:
            self.step()

    def step(self):
        """
        run one optimization cycle
        - get future objects
        - recreate history dataframe
        - determine the number of candidates to request from the generator
        - pass history dataframe and candidate request to generator
        - submit candidates to evaluator
        """

        # query futures and measure how many are still active
        finished_futures, unfinished_futures = concurrent.futures.wait(
            self.futures,
            self.timeout,
            self.return_when
        )

        # calculate number of new candidates to generate
        if self.asynch:
            n_generate = self.evaluator.max_workers - len(unfinished_futures)
        else:
            n_generate = self.evaluator.max_workers

        # generate samples and submit to evaluator
        new_samples = self.generator.generate(self.history, n_generate)
        self._samples += new_samples
        self._futures += self.evaluator.submit(new_samples)

    def process_config(self, config):
        """process the config file and create the evaluator, vocs, generator objects"""
        pass

    @property
    def vocs(self):
        return self._vocs

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def generator(self):
        return self._generator

    @property
    def futures(self):
        return self._futures

    @property
    def history(self):
        return self.create_dataframe()

    def create_dataframe(self) -> pd.DataFrame:
        """collect results and status from futures list"""
        data = []
        for sample, future in zip(self._samples, self._futures):
            new_dict = {**sample}
            if future.done():
                new_dict.update({**future.result(), "done": True})
            else:
                new_dict.update({"done": False})

            data += [new_dict]

        return pd.DataFrame(data)

