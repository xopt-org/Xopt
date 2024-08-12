import concurrent
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import Field

from xopt.base import logger, Xopt
from xopt.evaluator import validate_outputs


class AsynchronousXopt(Xopt):
    _futures: Dict = {}
    _ix_last: int = 0
    _n_unfinished_futures: int = 0
    _input_data: DataFrame = DataFrame([])
    is_done: bool = Field(
        default=False, description="flag indicating that Xopt " "fininshed running"
    )

    def submit_data(
        self,
        input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
    ):
        """
        Submit data to evaluator and return futures indexed to internal futures list.

        Parameters
        ----------
            input_data: dataframe containing input data

        """

        if not isinstance(input_data, DataFrame):
            try:
                input_data = DataFrame(input_data)
            except ValueError:
                input_data = DataFrame(input_data, index=[0])

        logger.debug(f"Submitting {len(input_data)} inputs")
        input_data = self.prepare_input_data(input_data)

        # submit data to evaluator. Futures are keyed on the index of the input data.
        futures = self.evaluator.submit_data(input_data)
        index = input_data.index

        # Special handling for vectorized evaluations
        if self.evaluator.vectorized:
            assert len(futures) == 1
            new_futures = {tuple(index): futures[0]}
        else:
            new_futures = dict(zip(index, futures))

        # add futures to internal list
        for key, future in new_futures.items():
            assert key not in self._futures, f"{key}, {self._futures}, {future}"
            self._futures[key] = future

        return futures

    def prepare_input_data(self, input_data: pd.DataFrame):
        """
        re-index and validate input data.
        """
        input_data = pd.DataFrame(input_data, copy=True)  # copy for reindexing

        # add constants to input data
        for name, value in self.vocs.constants.items():
            input_data[name] = value

        # Reindex input dataframe
        input_data.index = np.arange(
            self._ix_last + 1, self._ix_last + 1 + len(input_data)
        )
        self._ix_last += len(input_data)
        self._input_data = pd.concat([self._input_data, input_data])

        # validate data before submission
        self.vocs.validate_input_data(self._input_data)

        return input_data

    def step(self):
        if self.is_done:
            logger.debug("Xopt is done, will not step.")
            return

        # get number of candidates to generate
        n_generate = self.evaluator.max_workers - self._n_unfinished_futures

        # generate samples and submit to evaluator
        logger.debug(f"Generating {n_generate} candidates")
        new_samples = pd.DataFrame(self.generator.generate(n_generate))

        # Submit data
        self.submit_data(new_samples)
        # Process futures
        self._n_unfinished_futures = self.process_futures()

    def process_futures(self):
        logger.debug("Waiting for at least one future to complete")
        return_when = concurrent.futures.FIRST_COMPLETED

        # wait for futures to finish (depending on return_when)
        finished_futures, unfinished_futures = concurrent.futures.wait(
            self._futures.values(), None, return_when
        )

        # Get done indexes.
        ix_done = [ix for ix, future in self._futures.items() if future.done()]

        # Get results from futures
        output_data = []
        for ix in ix_done:
            future = self._futures.pop(ix)  # remove from futures
            outputs = future.result()  # Exceptions are already handled by the evaluator
            if self.strict:
                if future.exception() is not None:
                    raise future.exception()
                validate_outputs(pd.DataFrame(outputs, index=[1]))
            output_data.append(outputs)

        # Special handling of a vectorized futures.
        # Dict keys have all indexes of the input data.
        if self.evaluator.vectorized:
            output_data = pd.concat([pd.DataFrame([output]) for output in output_data])
            index = []
            for ix in ix_done:
                index.extend(list(ix))
        else:
            index = ix_done

        # Collect done inputs and outputs
        input_data_done = self._input_data.loc[index]
        output_data = pd.DataFrame(output_data, index=index)

        # Form completed evaluation
        new_data = pd.concat([input_data_done, output_data], axis=1)

        self.add_data(new_data)

        # Cleanup
        self._input_data.drop(index, inplace=True)

        return len(unfinished_futures)
