import concurrent
import threading
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import Field

from xopt.base import logger, Xopt
from xopt.errors import DataError
from xopt.evaluator import validate_outputs
from xopt.vocs import validate_input_data


class AsynchronousXopt(Xopt):
    _futures: Dict = None  # Will be initialized in __init__
    _ix_last: int = 0
    _n_unfinished_futures: int = 0
    _input_data: DataFrame = None  # Will be initialized in __init__
    _data_lock: threading.Lock = None  # Will be created lazily
    _global_index_counter: int = 0  # Global counter for unique indices
    is_done: bool = Field(
        default=False, description="flag indicating that Xopt fininshed running"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize instance-specific mutable objects
        self._futures = {}
        self._input_data = pd.DataFrame([])
        self._global_index_counter = 0

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
        for name, ele in self.vocs.constants.items():
            input_data[name] = ele.value

        # Reindex input dataframe
        input_data.index = np.arange(self._ix_last, self._ix_last + len(input_data))
        self._ix_last += len(input_data)
        self._input_data = pd.concat([self._input_data, input_data])

        # validate data before submission
        validate_input_data(self.vocs, self._input_data)

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

                try:
                    validate_outputs(pd.DataFrame(outputs))
                except ValueError:  # handle case where outputs is a dict of lists instead of list of dicts
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

    def add_data(self, new_data: pd.DataFrame):
        """
        Thread-safe version of add_data for concurrent access with guaranteed unique indices.

        Concatenate new data to the internal DataFrame and add it to the generator's
        data with proper synchronization to prevent race conditions and duplicate indices.

        Parameters
        ----------
        new_data : pd.DataFrame
            New data to be added to the internal DataFrame.
        """
        logger.debug(f"Adding {len(new_data)} new data to internal dataframes")

        with self.data_lock:
            # Set internal dataframe with thread safety and guaranteed unique indices
            if self.data is not None:
                new_data = pd.DataFrame(new_data, copy=True)  # copy for reindexing

                # Use global counter to ensure unique indices
                start_idx = self._global_index_counter
                new_data.index = np.arange(start_idx, start_idx + len(new_data))
                self._global_index_counter += len(new_data)

                # Double-check for uniqueness before concatenation
                if self.data.index.intersection(new_data.index).size > 0:
                    logger.warning(
                        "Detected potential index collision, regenerating indices"
                    )
                    # Fallback: use the actual max index + 1
                    max_existing_idx = (
                        self.data.index.max() if len(self.data) > 0 else -1
                    )
                    new_data.index = np.arange(
                        max_existing_idx + 1, max_existing_idx + 1 + len(new_data)
                    )
                    self._global_index_counter = max_existing_idx + 1 + len(new_data)

                self.data = pd.concat([self.data, new_data], axis=0)

                # Final validation: ensure no duplicate indices
                if not self.data.index.is_unique:
                    logger.error(
                        "Duplicate indices detected after concatenation, fixing..."
                    )
                    self.data = self.data.reset_index(drop=True)
                    self._global_index_counter = len(self.data)

            else:
                new_data = pd.DataFrame(new_data, copy=True)
                if new_data.index.dtype != np.int64:
                    new_data.index = new_data.index.astype(np.int64)
                # Ensure starting indices are sequential from 0
                new_data.index = np.arange(len(new_data))
                self._global_index_counter = len(new_data)
                self.data = new_data

        # Pass data to generator outside of lock to avoid potential deadlocks
        # Continue in case of invalid data when strict=False
        try:
            self.generator.ingest(new_data.to_dict(orient="records"))
        except DataError as exc:
            if self.strict:
                raise exc

    @property
    def data_lock(self):
        """Lazy initialization of the data lock to avoid pickling issues."""
        if self._data_lock is None:
            self._data_lock = threading.Lock()
        return self._data_lock

    def __getstate__(self):
        """Custom pickle method to exclude non-picklable threading objects."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        state["_data_lock"] = None
        # Remove futures as they are also not picklable
        state["_futures"] = {}
        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore state."""
        self.__dict__.update(state)
        # The lock will be recreated lazily when accessed
