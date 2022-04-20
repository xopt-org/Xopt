import numpy as np
import concurrent
import logging

import pandas as pd
from xopt.generators.generator import Generator
from .evaluator import Evaluator
from .vocs import VOCS

import traceback

logger = logging.getLogger(__name__)


class XoptBase:
    """

    Object to handle a single optimization problem.

    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        vocs: VOCS,
        asynch=False,
        timeout=None,
    ):
        # initialize XoptBase object
        self._generator = generator
        self._evaluator = evaluator
        self._vocs = vocs
        self.asynch = asynch
        self.timeout = timeout

        self._data = None
        self._futures = {}  # unfinished futures
        self._input_data = None  # dataframe for unfinished futures inputs
        self._ix_last = -1  # index of last sample generated
        self._is_done = False

        if self.asynch:
            self.return_when = concurrent.futures.FIRST_COMPLETED
        else:
            self.return_when = concurrent.futures.ALL_COMPLETED

    def run(self):
        """run until either xopt is done or the generator is done"""
        while not self._is_done:
            self.step()

    def submit_data(self, input_data: pd.DataFrame):

        input_data = pd.DataFrame(input_data, copy=True)  # copy for reindexing

        # Reindex input dataframe
        input_data.index = np.arange(
            self._ix_last + 1, self._ix_last + 1 + len(input_data)
        )
        self._ix_last += len(input_data)
        self._input_data = pd.concat([self._input_data, input_data])

        # submit data to evaluator. Futures are keyed on the index of the input data.
        futures = self.evaluator.submit_data(input_data)
        self._futures.update(futures)

    def step(self):
        """
        run one optimization cycle
        - get current set of future objects
        - determine the number of candidates to request from the generator
        - pass history dataframe and candidate request to generator
        - submit candidates to evaluator
        """
        if self._futures:
            # wait for futures to finish (depending on return_when)
            _, unfinished_futures = concurrent.futures.wait(
                self._futures.values(), self.timeout, self.return_when
            )

            # update dataframe with results from finished futures
            self.update_data()

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
        new_samples = pd.DataFrame(self.generator.generate(n_generate))

        # submit new samples to evaluator
        self.submit_data(new_samples)

    @property
    def data(self):
        return self._data

    @property
    def vocs(self):
        return self._vocs

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def generator(self):
        return self._generator

    def update_data(self, raises=False):
        # Get done indexes.
        ix_done = [ix for ix, future in self._futures.items() if future.done()]

        # Collect done inputs
        input_data_done = self._input_data.loc[ix_done]

        output_data = []
        for ix in ix_done:
            future = self._futures.pop(ix)  # remove from futures

            # Handle exceptions
            try:
                outputs = future.result()
                outputs["xopt_error"] = False
                outputs["xopt_error_str"] = ""
            except Exception as e:
                error_str = traceback.format_exc()
                outputs = {"xopt_error": True, "xopt_error_str": error_str}

            output_data.append(outputs)
        output_data = pd.DataFrame(output_data, index=ix_done)

        # Form completed evaluation
        new_data = pd.concat([input_data_done, output_data], axis=1)

        # Add to internal dataframe
        self._data = pd.concat([self._data, new_data], axis=0)

        # add to generator data
        self.generator.data = self.data

        # Cleanup
        self._input_data.drop(ix_done, inplace=True)

        # Return for convenience
        return new_data


0
