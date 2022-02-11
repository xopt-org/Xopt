import concurrent
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Xopt:
    """
    
    Object to handle a single optimization problem.
    
    Parameters
    ----------
    config: dict, YAML text, JSON text
        input file should be a dict, JSON, or YAML file with top level keys
    
          
    """
    _evaluator = None
    _generator = None
    _futures = []
    _history = None
    _vocs = {}
    _is_done = False
    asynch = False
    timeout = 1.0

    def __init__(self, config=None):
        # initialize Xopt object
        self.process_config(config)

        if self.asynch:
            self.return_when = concurrent.futures.FIRST_COMPLETED
        else:
            self.return_when = concurrent.futures.ALL_COMPLETED

    def run(self):
        """run until either xopt is done"""
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

        self._history = self.create_dataframe(self.futures)

        # calculate number of new candidates to generate
        if self.asynch:
            n_generate = self.evaluator.max_workers - len(unfinished_futures)
        else:
            n_generate = self.evaluator.max_workers

        # generate samples and submit to evaluator
        new_samples = self.generator.generate(self.history, n_generate)
        self._futures += [self.evaluator.submit(*new_samples)]

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
        return self._history

    @classmethod
    def create_dataframe(cls, futures) -> pd.DataFrame:
        """collect results and status from futures list"""
        pass


