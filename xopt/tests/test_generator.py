import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from xopt.generator import Generator
from xopt.generators import (
    generators,
    get_generator,
    get_generator_defaults,
    try_load_all_generators,
)
from xopt.resources.testing import TEST_VOCS_BASE


# Generators must be loaded to have access to names in generator tests
try_load_all_generators()


class TestGenerator:
    @patch.multiple(Generator, __abstractmethods__=set())
    def test_init(self):
        Generator(vocs=TEST_VOCS_BASE)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives.update({"y2": "MINIMIZE"})

        with pytest.raises(ValidationError):
            Generator(vocs=test_vocs)

    @pytest.mark.parametrize("name", list(generators.keys()))
    def test_serialization_loading(self, name):
        gen_config = get_generator_defaults(name)
        gen_class = get_generator(name)

        if name in ["mobo", "cnsga", "mggpo"]:
            gen_config["reference_point"] = {"y1": 10.0}
            json.dumps(gen_config)

            gen_class(vocs=TEST_VOCS_BASE, **gen_config)
        elif name in ["multi_fidelity"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        elif name in ["bayesian_exploration"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives = {}
            test_vocs.observables = ["f"]
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        else:
            json.dumps(gen_config)
            gen_class(vocs=TEST_VOCS_BASE, **gen_config)
