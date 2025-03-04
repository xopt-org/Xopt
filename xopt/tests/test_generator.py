import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from xopt.generator import Generator
from xopt.generators import generators, get_generator, get_generator_defaults
from xopt.resources.testing import TEST_VOCS_BASE


class TestGenerator:
    @patch.multiple(Generator, __abstractmethods__=set())
    def test_init(self):
        Generator(vocs=TEST_VOCS_BASE)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives.update({"y2": "MINIMIZE"})

        with pytest.raises(ValidationError):
            Generator(vocs=test_vocs)

    @pytest.mark.parametrize("generator_name", list(generators.keys()))
    def test_serialization_loading(self, generator_name):
        # Get generator defaults and class for this generator
        gen_config = get_generator_defaults(generator_name)
        gen_class = get_generator(generator_name)
        
        # Handle special cases for different generator types
        if generator_name in ["mobo", "cnsga", "mggpo"]:
            gen_config["reference_point"] = {"y1": 10.0}
            test_vocs = TEST_VOCS_BASE
        elif generator_name in ["multi_fidelity"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
        elif generator_name in ["bayesian_exploration"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives = {}
            test_vocs.observables = ["f"]
        else:
            test_vocs = TEST_VOCS_BASE
        
        # Test JSON serialization
        serialized = json.dumps(gen_config)
        assert serialized is not None
        
        # Test instantiation with the configuration
        generator = gen_class(vocs=test_vocs, **gen_config)
        assert generator is not None