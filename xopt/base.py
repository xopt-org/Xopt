import numpy as np
import concurrent
import logging

import pandas as pd
import yaml

from xopt.generator import Generator
from xopt.evaluator import Evaluator
from xopt.io import read_config_dict, state_to_dict, load_state_yaml
from xopt.vocs import VOCS
from xopt.errors import XoptError
from xopt.options import XoptOptions

import traceback

logger = logging.getLogger(__name__)


class Xopt:
    """

    Object to handle a single optimization problem.

    """

    def __init__(
        self,
        *,
        config: dict = None,
        generator: Generator = None,
        evaluator: Evaluator = None,
        vocs: VOCS = None,
        options: XoptOptions = XoptOptions()
    ):
        """
        Initialize Xopt object

        Args:
            config: dict containing configuration information
            generator: Generator object
            evaluator: Evaluator object
            vocs: VOCS object
            options: XoptOptions object

        """
        # if config is provided, load it
        data = None
        if config is not None:
            generator, evaluator, vocs, options, data = load_state_yaml(config)

        # initialize Xopt object
        self._generator = generator
        self._evaluator = evaluator
        self._vocs = vocs

        # do options
        if not isinstance(options, XoptOptions):
            raise ValueError("options must of type `XoptOptions`")

        self.options = options

        self._data = data if data is not None else pd.DataFrame()
        self._new_data = None
        self._futures = {}  # unfinished futures
        self._input_data = None  # dataframe for unfinished futures inputs
        self._ix_last = len(self._data)  # index of last sample generated
        self._is_done = False
        self.n_unfinished_futures = 0


        # check internals
        self.check_components()

    def run(self):
        """run until either xopt is done or the generator is done"""
        while not self._is_done:
            self.step()

    def submit_data(self, input_data: pd.DataFrame):
        """
        Submit data to evaluator and append results to internal futures list

        """
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

        - determine the number of candidates to request from the generator
        - pass candidate request to generator
        - submit candidates to evaluator
        - wait until all (asynch == False) or at least one (asynch == True) evaluation
            is finished
        - update data storage and generator data storage (if applicable)

        """

        # get number of candidates to generate
        if self.options.asynch:
            n_generate = self.evaluator.max_workers - self.n_unfinished_futures
        else:
            n_generate = self.evaluator.max_workers

        # generate samples and submit to evaluator
        new_samples = pd.DataFrame(self.generator.generate(n_generate))

        # submit new samples to evaluator
        self.submit_data(new_samples)

        # process futures after waiting for one or all to be completed
        # get number of uncompleted futures when done waiting
        self.n_unfinished_futures = self.process_futures()

    def process_futures(self):
        """
        wait for futures to finish (specified by asynch) and then internal dataframes
        of xopt and generator, finally return the number of unfinished futures

        """
        if self.options.asynch:
            return_when = concurrent.futures.FIRST_COMPLETED
        else:
            return_when = concurrent.futures.ALL_COMPLETED

        # wait for futures to finish (depending on return_when)
        finished_futures, unfinished_futures = concurrent.futures.wait(
            self._futures.values(), self.options.timeout, return_when
        )

        # if strict, raise exception if any future raises an exception
        if self.options.strict:
            for f in finished_futures:
                if f.exception() is not None:
                    raise f.exception()

        # update dataframe with results from finished futures + generator data
        self.update_data()

        # dump data to file if specified
        self.dump_state()

        return len(unfinished_futures)

    def update_data(self):
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

        # Add to internal dataframes
        self._data = pd.concat([self._data, new_data], axis=0)
        self._new_data = new_data

        # The generator can optionally use new data
        self.generator.add_data(self._new_data)

        # Cleanup
        self._input_data.drop(ix_done, inplace=True)

    def check_components(self):
        """check to make sure everything is in place to step"""
        if self.generator is None:
            raise XoptError("Xopt generator not specified")

        if self.evaluator is None:
            raise XoptError("Xopt evaluator not specified")

        if self.vocs is None:
            raise XoptError("Xopt VOCS is not specified")

    def dump_state(self):
        if self.options.dump_file is not None:
            output = state_to_dict(self)
            with open(self.options.dump_file, "w") as f:
                yaml.dump(output, f)

    @property
    def data(self):
        return self._data

    @property
    def new_data(self):
        return self._new_data

    @property
    def vocs(self):
        return self._vocs

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def generator(self):
        return self._generator
