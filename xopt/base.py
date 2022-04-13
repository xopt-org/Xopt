import concurrent
import logging
from typing import Type, List, Dict

import pandas as pd
from .generator import Generator
from .evaluator import Evaluator
from .vocs import VOCS
from .utils import add_constraint_information

logger = logging.getLogger(__name__)


class XoptBase:
    """

    Object to handle a single optimization problem.

    """

    _futures = pd.Series()
    _is_done = False
    timeout = 1.0

    def __init__(
            self, generator: Generator, evaluator: Evaluator, vocs: VOCS, asynch=False
    ):
        # initialize XoptBase object
        self._generator = generator
        self._evaluator = evaluator
        self._vocs = vocs
        self.asynch = asynch

        if self.asynch:
            self.return_when = concurrent.futures.FIRST_COMPLETED
        else:
            self.return_when = concurrent.futures.ALL_COMPLETED

        # initialize dataframe
        self.data = pd.DataFrame(columns=self.vocs.all_names)

    def run(self):
        """run until either xopt is done or the generator is done"""
        while not self._is_done:
            self.step()

    def submit(self, samples: pd.DataFrame):
        """
        submit a list of dicts to the evaluator
        (also appends submissions to futures attribute)

        """
        self._futures = pd.concat([self._futures, self.evaluator.submit(samples)])
        self.data = pd.concat([self.data, samples])

    def step(self):
        """
        run one optimization cycle
        - get current set of future objects
        - determine the number of candidates to request from the generator
        - pass history dataframe and candidate request to generator
        - submit candidates to evaluator
        """
        if not self._futures.empty:
            # wait for futures to finish (depending on return_when)
            _, unfinished_futures = concurrent.futures.wait(
                self._futures, self.timeout, self.return_when
            )

            # update dataframe with results from finished futures
            self.update_dataframe()

            # calculate number of new candidates to generate
            if self.asynch:
                n_generate = self.evaluator.max_workers - len(unfinished_futures)
            else:
                n_generate = self.evaluator.max_workers
        else:
            n_generate = self.evaluator.max_workers

        # update data in generator
        self.generator.data = self.data

        # generate samples and submit to evaluator
        new_samples = self.generator.generate(n_generate)

        # modify index of new samples to align with old samples
        new_samples.index = new_samples.index + len(self.data)

        self.submit(new_samples)

    @property
    def vocs(self):
        return self._vocs

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def generator(self):
        return self._generator

    def update_dataframe(self):
        """
        update dataframe with results from finished futures
        """
        results = pd.DataFrame(
            list(self._futures.map(lambda x: x.result())),
            index=self._futures.index
        )

        self.data.update(results)
