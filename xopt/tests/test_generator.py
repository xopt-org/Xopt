import json
from copy import deepcopy
import pandas as pd
import pytest

from xopt.errors import VOCSError
from xopt.generator import Generator
from xopt.generators import (
    get_generator,
    get_generator_defaults,
    list_available_generators,
)
from xopt.resources.testing import TEST_VOCS_BASE
from gest_api.vocs import VOCS


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


class PatchGeneratorNoConstraints(Generator):
    name = "patch_generator_no_constraints"

    def generate(self, n_candidates) -> list[dict]:
        pass


class TestGenerator:
    def test_init(self):
        PatchGenerator(vocs=TEST_VOCS_BASE)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives.update({"y2": "MINIMIZE"})

        with pytest.raises(VOCSError):
            PatchGenerator(vocs=test_vocs)

    def test_add_data(self):
        gen = PatchGenerator(vocs=TEST_VOCS_BASE)
        data = [{"x1": 1.0, "x2": 2.0, "y1": 3.0}]
        gen.ingest(data)
        assert (gen.data == pd.DataFrame(data)).all().all()

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
        elif name in ["bayesian_exploration", "latin_hypercube"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives = {"f": "EXPLORE"}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        else:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)

    def test_generator_constraints_validation(self):
        vocs_with_constraints = VOCS(
            variables={"x1": [0, 1], "x2": [0, 1]},
            objectives={"y1": "MINIMIZE"},
            constraints={"c1": ["GREATER_THAN", 0.0]},
        )

        with pytest.raises(
            VOCSError, match="this generator does not support constraints"
        ):
            PatchGeneratorNoConstraints(vocs=vocs_with_constraints)

    def test_data_validator_indexerror(self):
        data_dict = {"x1": 1.0, "x2": 2.0}
        gen = PatchGenerator(vocs=TEST_VOCS_BASE, data=data_dict)
        # Should fallback to pd.DataFrame(data_dict, index=[0])
        assert isinstance(gen.data, pd.DataFrame)
        assert gen.data.shape[0] == 1
        assert set(gen.data.columns) == {"x1", "x2"}
        assert gen.data.iloc[0]["x1"] == 1.0
        assert gen.data.iloc[0]["x2"] == 2.0
