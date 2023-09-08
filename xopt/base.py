import json
import logging
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from pydantic import Field, validator

from xopt import _version
from xopt.evaluator import Evaluator, validate_outputs
from xopt.generator import Generator
from xopt.generators import get_generator
from xopt.pydantic import XoptBaseModel
from xopt.utils import explode_all_columns
from xopt.vocs import VOCS

__version__ = _version.get_versions()["version"]

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
    data: DataFrame = Field(None, description="internal DataFrame object")
    serialize_torch: bool = Field(
        False,
        description="flag to indicate that torch models should be serialized "
        "when dumping",
    )
    max_evaluations: int = Field(None)

    @validator("vocs", pre=True)
    def validate_vocs(cls, value):
        if isinstance(value, dict):
            value = VOCS(**value)
        return value

    @validator("evaluator", pre=True)
    def validate_evaluator(cls, value):
        if isinstance(value, dict):
            value = Evaluator(**value)

        return value

    @validator("generator", pre=True)
    def validate_generator(cls, value, values):
        if isinstance(value, dict):
            name = value.pop("name")
            generator_class = get_generator(name)
            value = generator_class.parse_obj({**value, "vocs": values["vocs"]})
        elif isinstance(value, str):
            generator_class = get_generator(value)
            value = generator_class.parse_obj({"vocs": values["vocs"]})

        return value

    @validator("data", pre=True)
    def validate_data(cls, v):
        if isinstance(v, dict):
            try:
                v = pd.DataFrame(v)
            except IndexError:
                v = pd.DataFrame(v, index=[0])
        return v

    @property
    def n_data(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

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
        new_samples = self.generator.generate(n_generate)

        # Evaluate data
        self.evaluate_data(new_samples)

    def run(self):
        """run until either max_evaluations is reached or the generator is
        done"""
        while not self.generator.is_done:
            # Stopping criteria
            if self.max_evaluations is not None:
                if self.n_data >= self.max_evaluations:
                    logger.info(
                        "Xopt is done. "
                        f"Max evaluations {self.max_evaluations} reached."
                    )
                    break

            self.step()

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
        new_data = explode_all_columns(new_data)

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

        # Set internal dataframe.
        if self.data is not None:
            new_data = pd.DataFrame(new_data, copy=True)  # copy for reindexing
            new_data.index = np.arange(
                len(self.data) + 1, len(self.data) + len(new_data) + 1
            )

            self.data = pd.concat([self.data, new_data], axis=0)
        else:
            self.data = new_data
        self.generator.add_data(new_data)

    def reset_data(self):
        self.data = pd.DataFrame()
        self.generator.data = pd.DataFrame()

    def random_evaluate(self, n_samples=1, seed=None, **kwargs):
        """
        Convenience method to generate random inputs using vocs
        and evaluate them (adding data to Xopt object and generator).
        """
        random_inputs = self.vocs.random_inputs(n_samples, seed=seed, **kwargs)
        result = self.evaluate_data(random_inputs[self.vocs.variable_names])
        return result

    def dump_state(self):
        """dump data to file"""
        if self.dump_file is not None:
            output = json.loads(self.json(serialize_torch=self.serialize_torch))
            with open(self.dump_file, "w") as f:
                yaml.dump(output, f)
            logger.debug(f"Dumped state to YAML file: {self.dump_file}")

    def dict(self, **kwargs) -> Dict:
        """handle custom dict generation"""
        result = super().dict(**kwargs)
        result["generator"] = {"name": self.generator.name} | result["generator"]
        return result

    def json(self, **kwargs) -> str:
        """handle custom serialization of generators and dataframes"""
        result = super().json(**kwargs)
        dict_result = json.loads(result)
        dict_result["generator"] = {"name": self.generator.name} | dict_result[
            "generator"
        ]
        dict_result["data"] = json.loads(self.data.to_json())

        # TODO: implement version checking
        # dict_result["xopt_version"] = __version__

        return json.dumps(dict_result)

    def __repr__(self):
        """
        Returns infor about the Xopt object, including the YAML representation without data.
        """

        # get dict minus data
        config = deepcopy(self.dict())
        config.pop("data")
        return f"""
            Xopt
________________________________
Version: {__version__}
Data size: {self.n_data}
Config as YAML:
{yaml.dump(config)}
"""

    def __str__(self):
        return self.__repr__()
