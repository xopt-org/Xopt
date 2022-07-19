import json
from copy import deepcopy

from pydantic import Field

from xopt import _version
from xopt.errors import XoptError
from xopt.evaluator import Evaluator
from xopt.generator import Generator
from xopt.pydantic import XoptBaseModel
from xopt.utils import get_generator_and_defaults
from xopt.vocs import VOCS

__version__ = _version.get_versions()["version"]

import concurrent
import logging
import os
import traceback

from typing import Dict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class XoptOptions(XoptBaseModel):
    asynch: bool = Field(
        False, description="flag to evaluate and submit evaluations asynchronously"
    )
    strict: bool = Field(
        False,
        description="flag to indicate if exceptions raised during evaluation "
        "should stop Xopt",
    )
    dump_file: str = Field(
        None, description="file to dump the results of the evaluations"
    )
    max_evaluations: int = Field(
        None, description="maximum number of evaluations to perform"
    )


class Xopt:
    """

    Object to handle a single optimization problem.

    """

    def __init__(
        self,
        config: dict = None,
        *,
        generator: Generator = None,
        evaluator: Evaluator = None,
        vocs: VOCS = None,
        options: XoptOptions = XoptOptions(),
        data: pd.DataFrame = None,
    ):
        """
        Initialize Xopt object using either a config dictionary or explicitly

        Args:
            config: dict, or YAML or JSON str or file. This overrides all other arguments.

            generator: Generator object
            evaluator: Evaluator object
            vocs: VOCS object
            options: XoptOptions object
            data: initial data to use

        """
        logger.info("Initializing Xopt object")

        # if config is provided, load it and re-init. Otherwise, init normally.
        if config is not None:
            self.__init__(**parse_config(config))
            # TODO: Allow overrides
            return

        # initialize Xopt object
        self._generator = generator
        self._evaluator = evaluator
        self._vocs = vocs

        logger.debug(f"Xopt initialized with generator: {self._generator}")
        logger.debug(f"Xopt initialized with evaluator: {self._evaluator}")

        if not isinstance(options, XoptOptions):
            raise ValueError("options must of type `XoptOptions`")
        self.options = options
        logger.debug(f"Xopt initialized with options: {self.options.dict()}")

        self._data = pd.DataFrame(data)
        self._new_data = None
        self._futures = {}  # unfinished futures
        self._input_data = None  # dataframe for unfinished futures inputs
        self._ix_last = len(self._data)  # index of last sample generated
        self._is_done = False
        self.n_unfinished_futures = 0

        # check internals
        self.check_components()

        logger.info("Xopt object initialized")

    def run(self):
        """run until either xopt is done or the generator is done"""
        while not self.is_done:

            # Stopping criteria
            if self.options.max_evaluations:
                if len(self.data) >= self.options.max_evaluations:
                    self._is_done = True
                    logger.info(
                        f"Xopt is done. Max evaluations {self.options.max_evaluations} reached."
                    )
                    break

            self.step()

    def submit_data(self, input_data: pd.DataFrame):
        """
        Submit data to evaluator and append results to internal dataframe.

        Args:
            input_data: dataframe containing input data

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

        # wait for futures
        self.wait_for_futures()

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
        logger.info("Running Xopt step")
        if self.is_done:
            logger.debug('Xopt is done, will not step.')
            return

        # get number of candidates to generate
        if self.options.asynch:
            n_generate = self.evaluator.max_workers - self.n_unfinished_futures
        else:
            n_generate = self.evaluator.max_workers

        # generate samples and submit to evaluator
        logger.debug(f"Generating {n_generate} candidates")
        new_samples = pd.DataFrame(self.generator.generate(n_generate))

        # generator is done when it returns no new samples
        if len(new_samples) == 0:
            logger.debug('Generator returned 0 samples => optimization is done.')
            assert self.generator.is_done
            return

        # submit new samples to evaluator
        logger.debug(f"Submitting {len(new_samples)} candidates to evaluator")
        self.submit_data(new_samples)

    def wait_for_futures(self):
        # process futures after waiting for one or all to be completed
        # get number of uncompleted futures when done waiting
        if self.options.asynch:
            logger.debug("Waiting for at least one future to complete")
        else:
            logger.debug("Waiting for all futures to complete")

        self.n_unfinished_futures = self.process_futures()
        logger.debug(f"done. {self.n_unfinished_futures} futures remaining")

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
            self._futures.values(), None, return_when
        )

        # if strict, raise exception if any future raises an exception
        if self.options.strict:
            for f in finished_futures:
                if f.exception() is not None:
                    raise f.exception()

        # update dataframe with results from finished futures + generator data
        logger.debug("Updating dataframes with results")
        self.update_data()

        # dump data to file if specified
        self.dump_state()

        return len(unfinished_futures)

    def update_data(self):
        """
        process finished futures and update dataframes

        Dataframes are updated in the following order:
        - xopt._data
        - xopt._new_data

        Data is added to the dataframe in the following order:
        - generator._data

        """
        # Get done indexes.
        ix_done = [ix for ix, future in self._futures.items() if future.done()]

        # Collect done inputs
        input_data_done = self._input_data.loc[ix_done]

        output_data = []
        for ix in ix_done:
            future = self._futures.pop(ix)  # remove from futures

            # Handle exceptions inside the future result
            try:
                # if the future raised an exception, it gets raised here
                outputs = future.result()

                # check to make sure the output is a dict - if so raise XoptError
                if not isinstance(outputs, dict):
                    raise XoptError(
                        f"Evaluator function must return a dict, got type "
                        f"{type(future.result())} instead"
                    )

                outputs["xopt_error"] = False
                outputs["xopt_error_str"] = ""
            except Exception as e:

                # if the exception is an XoptError, re-raise it
                if isinstance(e, XoptError):
                    raise e

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
        """dump data to file"""
        if self.options.dump_file is not None:
            output = state_to_dict(self)
            with open(self.options.dump_file, "w") as f:
                yaml.dump(output, f)
            logger.debug(f"Dumping state to:{self.options.dump_file}")

    @property
    def data(self):
        return self._data

    @property
    def is_done(self):
        return self._is_done or self.generator.is_done

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

    @classmethod
    def from_dict(cls, config_dict):
        pass
        # return cls(**xopt_kwargs_from_dict(config_dict))

    @classmethod
    def from_yaml(cls, yaml_str):
        if os.path.exists(yaml_str):
            yaml_str = open(yaml_str)
        return cls.from_dict(yaml.safe_load(yaml_str))

    def yaml(self, filename=None, *, include_data=False):
        """
        YAML representation of the Xopt object.
        """
        config = state_to_dict(self, include_data=include_data)
        s = yaml.dump(config, default_flow_style=None, sort_keys=False)

        if filename:
            with open(filename, "w") as f:
                f.write(s)

        return s

    def __repr__(self):
        """
        Returns infor about the Xopt object, including the YAML representation without data.
        """
        return f"""
            Xopt
________________________________
Version: {__version__}
Data size: {len(self.data)}
Config as YAML:
{self.yaml()}
"""

    def __str__(self):
        return self.__repr__()

    # Convenience methods

    def random_inputs(self, *args, **kwargs):
        """
        Convenence method to call vocs.random_inputs
        """
        return self.vocs.random_inputs(*args, **kwargs)

    def evaluate(self, inputs: Dict, **kwargs):
        """
        Convenience method to call evaluator.evaluate
        """
        return self.evaluator.evaluate(inputs, **kwargs)

    def random_evaluate(self, *args, **kwargs):
        """
        Convenience method to generate random inputs using vocs
        and evaluate them using evaluator.evaluate.
        """
        result = self.evaluate(self.random_inputs(*args, **kwargs))
        return result


def parse_config(config) -> dict:
    """
    Parse a config, which can be:
        YAML file
        JSON file
        dict-like object

    Returns a dict of kwargs for Xopt constructor.
    """
    if isinstance(config, str):
        if os.path.exists(config):
            yaml_str = open(config)
        else:
            yaml_str = config
        d = yaml.safe_load(yaml_str)
    else:
        d = config

    return xopt_kwargs_from_dict(d)


def xopt_kwargs_from_dict(config: dict) -> dict:
    """
    Processes a config dictionary and returns the corresponding Xopt kwargs.
    """

    # get copy of config
    config = deepcopy(config)

    options = XoptOptions(**config["xopt"])
    vocs = VOCS(**config["vocs"])

    # create generator
    generator_type, generator_options = get_generator_and_defaults(
        config["generator"].pop("name")
    )
    # TODO: use version number in some way
    if "version" in config["generator"].keys():
        config["generator"].pop("version")

    generator = generator_type(vocs, generator_options.parse_obj(config["generator"]))

    # Create evaluator
    evaluator = Evaluator(**config["evaluator"])

    # OldEvaluator
    # ev = config["evaluator"]
    # ev["function"] = get_function(ev["function"])
    # ev_options = EvaluatorOptions.parse_obj(ev)
    # evaluator = Evaluator(**ev_options.dict())

    if "data" in config.keys():
        data = config["data"]
    else:
        data = None

    # return generator, evaluator, vocs, options, data
    return {
        "generator": generator,
        "evaluator": evaluator,
        "vocs": vocs,
        "options": options,
        "data": data,
    }


def state_to_dict(X, include_data=True):
    # dump data to dict with config metadata
    output = {
        "xopt": json.loads(X.options.json()),
        "generator": {
            "name": X.generator.alias,
            **json.loads(X.generator.options.json()),
        },
        "evaluator": json.loads(X.evaluator.json()),
        "vocs": json.loads(X.vocs.json()),
    }
    if include_data:
        output["data"] = json.loads(X.data.to_json())

    return output
