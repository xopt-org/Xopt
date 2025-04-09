import json
from copy import deepcopy

import pytest

from xopt.errors import VOCSError
from xopt.generator import Generator
from xopt.generators import (
    get_generator,
    get_generator_defaults,
    list_available_generators,
)
from xopt.resources.testing import TEST_VOCS_BASE


class PatchGenerator(Generator):
    """
    Test generator class for testing purposes.
    """

    name = "test_generator"
    supports_batch_generation: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True

    def generate(self, n_candidates) -> list[dict]:
        pass


class TestGenerator:
    def test_init(self):
        PatchGenerator(vocs=TEST_VOCS_BASE)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives.update({"y2": "MINIMIZE"})

        with pytest.raises(VOCSError):
            PatchGenerator(vocs=test_vocs)

    @pytest.mark.parametrize("name", list_available_generators())
    def test_serialization_loading(self, name):
        gen_config = get_generator_defaults(name)
        gen_class = get_generator(name)

        if name in ["mobo", "cnsga", "mggpo"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives.update({"y2": "MINIMIZE"})
            gen_config["reference_point"] = {"y1": 10.0, "y2": 1.5}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        elif name in ["nsga2"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives.update({"y2": "MINIMIZE"})
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)

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
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
