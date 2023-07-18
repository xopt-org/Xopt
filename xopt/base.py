import json
from typing import Dict

from pandas import DataFrame
from pydantic import Field, validator

from xopt import _version
from xopt.evaluator import Evaluator, validate_outputs
from xopt.generator import Generator
from xopt.generators import get_generator
from xopt.pydantic import XoptBaseModel
from xopt.vocs import VOCS

__version__ = _version.get_versions()["version"]

import logging

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class Xopt(XoptBaseModel):
    """
    Object to handle a single optimization problem.
    """
    vocs: VOCS = Field(description="VOCS object for Xopt")
    generator: Generator = Field(description="generator object for Xopt")
    evaluator: Evaluator = Field(description="evaluator object for Xopt")

    strict: bool = Field(
        True,
        description="flag to indicate if exceptions raised during evaluation "
        "should stop Xopt",
    )
    dump_file: str = Field(
        None, description="file to dump the results of the evaluations"
    )
    data: DataFrame = Field(
        pd.DataFrame(), description="internal DataFrame object"
    )

    @validator("vocs", pre=True)
    def validate_vocs(cls, value):
        if isinstance(value, VOCS):
            return value
        elif isinstance(value, dict):
            return VOCS(**value)

    @validator("evaluator", pre=True)
    def validate_evaluator(cls, value):
        if isinstance(value, Evaluator):
            return value
        elif isinstance(value, dict):
            return Evaluator(**value)

    @validator("generator", pre=True)
    def validate_generator(cls, value, values):
        if isinstance(value, Generator):
            return value
        elif isinstance(value, dict):
            name = value.pop("name")
            generator_class = get_generator(name)
            return generator_class.parse_obj(
                {**value, "vocs": values["vocs"]})
        elif isinstance(value, str):
            generator_class = get_generator(value)
            return generator_class.parse_obj(
                {"vocs": values["vocs"]})

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

        # get number of candidates to generate
        n_generate = self.evaluator.max_workers

        # generate samples and submit to evaluator
        logger.debug(f"Generating {n_generate} candidates")
        new_samples = pd.DataFrame(self.generator.generate(n_generate))

        # Evaluate data
        self.evaluate_data(new_samples)

    def evaluate_data(self, input_data: pd.DataFrame):
        """
        Evaluate data using the evaluator and wait for results.
        Adds results to the internal dataframe.
        """
        logger.debug(f"Evaluating {len(input_data)} inputs")
        self.vocs.validate_input_data(input_data)
        output_data = self.evaluator.evaluate_data(input_data)

        if self.strict:
            validate_outputs(output_data)
        new_data = pd.concat([input_data, output_data], axis=1)

        # explode any list like results if all of the output names exist
        try:
            new_data = new_data.explode(self.vocs.output_names)
        except KeyError:
            pass

        self.add_data(new_data)

        # dump data to file if specified
        self.dump_state()

        return new_data

    def add_data(self, new_data: pd.DataFrame):
        """
        Concatenate new data to internal dataframe,
        and also adds this data to the generator.
        """
        logger.debug(f"Adding {len(new_data)} new data to internal dataframes")

        # Set internal dataframe. Don't use self.data =
        new_data = pd.DataFrame(new_data, copy=True)  # copy for reindexing
        new_data.index = np.arange(
            len(self.data) + 1, len(self.data) + len(new_data) + 1
        )
        self.data = pd.concat([self.data, new_data], axis=0)
        self.generator.add_data(new_data)

    def reset_data(self):
        self.data = pd.DataFrame()
        self.generator.data = pd.DataFrame()

    def random_evaluate(self, n_samples=1, seed=None, **kwargs):
        """
        Convenience method to generate random inputs using vocs
        and evaluate them (adding data to Xopt object and generator.
        """
        index = [1] if n_samples == 1 else None
        random_inputs = pd.DataFrame(
            self.vocs.random_inputs(n_samples, seed=seed, **kwargs), index=index
        )
        result = self.evaluate_data(random_inputs)
        return result

    def dump_state(self):
        """dump data to file"""
        if self.dump_file is not None:
            output = json.loads(self.json())
            with open(self.dump_file, "w") as f:
                yaml.dump(output, f)
            logger.debug(f"Dumped state to YAML file: {self.dump_file}")

    def dict(self, **kwargs) -> Dict:
        """ handle custom dict generation"""
        result = super().dict(**kwargs)
        result["generator"] = {"name": self.generator.name} | result["generator"]
        return result

    def json(self, **kwargs) -> str:
        """ handle custom serialization of generators and dataframes"""
        result = super().json(**kwargs)
        dict_result = json.loads(result)
        dict_result["generator"] = {
            "name": self.generator.name
        } | dict_result["generator"]
        dict_result["data"] = json.loads(self.data.to_json())

        # TODO: implement version checking
        #dict_result["xopt_version"] = __version__

        return json.dumps(dict_result)
