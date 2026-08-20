import json
import logging
import os

import pandas as pd
import pytest

from xopt.generators.ga.base import GAGeneratorBase
from xopt.resources.test_functions.tnk import tnk_vocs


class OutputTestGenerator(GAGeneratorBase):
    """Minimal concrete generator for exercising the base class output behavior."""

    name = "ga_base_test"
    supports_single_objective: bool = True
    supports_multi_objective: bool = True
    supports_constraints: bool = True

    def _generate(self, n_candidates: int) -> list[dict]:
        return []


class RecordingHandler(logging.Handler):
    """Captures messages so propagation to the module logger can be checked."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def module_logger():
    """Handler on the logger records propagate to, named after the concrete class."""
    logger = logging.getLogger(OutputTestGenerator.__module__)
    handler = RecordingHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


def make_generator(output_dir, **kwargs):
    return OutputTestGenerator(
        vocs=tnk_vocs,
        output_dir=None if output_dir is None else str(output_dir),
        log_level=logging.DEBUG,
        **kwargs,
    )


def make_population(
    size: int, generation: int, extra: dict | None = None
) -> list[dict]:
    """Build a population of individuals carrying all VOCS and metadata columns."""
    population = []
    for idx in range(size):
        individual = {
            "x1": 0.1 * idx,
            "x2": 0.2 * idx,
            "y1": 1.0 * idx,
            "y2": 2.0 * idx,
            "c1": -1.0,
            "c2": -1.0,
            "xopt_candidate_idx": generation * size + idx,
            "xopt_runtime": 0.1,
            "xopt_error": False,
        }
        if extra is not None:
            individual.update(extra)
        population.append(individual)
    return population


def make_data(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.1 * i for i in range(n_rows)],
            "x2": [0.2 * i for i in range(n_rows)],
            "xopt_candidate_idx": list(range(n_rows)),
            "xopt_parent_generation": [0] * n_rows,
        }
    )


def run_generation(generator, index, size=4, extra=None, n_data=None):
    """Feed the generator a completed generation."""
    generator.data = make_data(n_data if n_data is not None else index * size)
    generator.end_generation(index, make_population(size, index - 1, extra))


def test_construction_touches_nothing(tmp_path):
    # Generators are built and deserialized freely, so neither may create anything
    requested = tmp_path / "run"
    generator = make_generator(requested)
    OutputTestGenerator.model_validate(json.loads(generator.to_json()))

    assert generator.output_dir == str(requested)
    assert os.listdir(tmp_path) == []


def test_prepare_output_creates_directory_and_is_idempotent(tmp_path):
    requested = tmp_path / "run"
    generator = make_generator(requested)

    generator._prepare_output()
    assert generator.output_dir == str(requested)
    assert os.path.isdir(requested)

    generator._prepare_output()
    assert generator.output_dir == str(requested)
    generator.close_log_file()


def test_existing_empty_directory_is_not_renamed(tmp_path, module_logger):
    # A directory which exists but holds nothing is reused as-is. Tests which hand
    # the generator a TemporaryDirectory depend on this.
    requested = tmp_path / "run"
    os.makedirs(requested)

    generator = make_generator(requested)
    generator._prepare_output()

    assert generator.output_dir == str(requested)
    assert not any("corrected" in m for m in module_logger.messages)
    generator.close_log_file()


def test_non_empty_directory_is_renamed(tmp_path, module_logger):
    requested = tmp_path / "run"
    os.makedirs(requested)
    (requested / "data.csv").write_text("existing\n")

    first = make_generator(requested)
    first._prepare_output()
    assert first.output_dir == f"{requested}_2"
    assert os.path.isdir(f"{requested}_2")
    assert any("corrected" in m for m in module_logger.messages)

    # The original directory is left untouched
    assert (requested / "data.csv").read_text() == "existing\n"
    first.close_log_file()

    # A second collision steps to the next suffix
    (tmp_path / "run_2" / "data.csv").write_text("existing\n")
    second = make_generator(requested)
    second._prepare_output()
    assert second.output_dir == f"{requested}_3"
    second.close_log_file()


def test_end_generation_writes_both_files(tmp_path):
    generator = make_generator(tmp_path / "run")
    run_generation(generator, 1, n_data=8)

    assert len(pd.read_csv(os.path.join(generator.output_dir, "data.csv"))) == 8

    pop_df = pd.read_csv(os.path.join(generator.output_dir, "populations.csv"))
    assert len(pop_df) == 4
    assert (pop_df["xopt_generation"] == 1).all()
    assert list(pop_df.columns) == tnk_vocs.all_names + [
        "xopt_generation",
        "xopt_candidate_idx",
        "xopt_runtime",
        "xopt_error",
    ]
    generator.close_log_file()


def test_data_overwritten_while_populations_accumulate(tmp_path):
    generator = make_generator(tmp_path / "run")
    run_generation(generator, 1, n_data=4)
    run_generation(generator, 2, n_data=8)

    # data.csv is a full overwrite, so it reflects only the latest generation
    assert len(pd.read_csv(os.path.join(generator.output_dir, "data.csv"))) == 8

    # populations.csv is appended and carries exactly one header line
    population_path = os.path.join(generator.output_dir, "populations.csv")
    pop_df = pd.read_csv(population_path)
    assert len(pop_df) == 8
    assert sorted(pop_df["xopt_generation"].unique()) == [1, 2]
    with open(population_path) as f:
        assert sum(1 for line in f if line.startswith("x1,")) == 1
    generator.close_log_file()


def test_end_generation_normalizes_changing_schema(tmp_path):
    generator = make_generator(tmp_path / "run")
    run_generation(generator, 1)

    # A later generation gaining an extra key must not shift the appended columns
    run_generation(generator, 2, extra={"obs1": 3.0})

    # ... nor may one missing a metadata key
    generator.data = make_data(12)
    sparse = make_population(4, 2)
    for individual in sparse:
        del individual["xopt_runtime"]
    generator.end_generation(3, sparse)

    pop_df = pd.read_csv(os.path.join(generator.output_dir, "populations.csv"))
    assert len(pop_df) == 12
    assert "obs1" not in pop_df.columns
    assert pop_df[pop_df["xopt_generation"] == 3]["xopt_runtime"].isna().all()
    assert pop_df[pop_df["xopt_generation"] == 2]["xopt_runtime"].notna().all()
    generator.close_log_file()


@pytest.mark.parametrize("checkpoint_freq, expected", [(1, 4), (2, 2), (-1, 0)])
def test_checkpoint_frequency(tmp_path, checkpoint_freq, expected):
    generator = make_generator(tmp_path / "run", checkpoint_freq=checkpoint_freq)
    for index in range(1, 5):
        run_generation(generator, index)

    checkpoint_dir = os.path.join(generator.output_dir, "checkpoints")
    written = len(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else 0
    assert written == expected
    generator.close_log_file()


def test_no_output_dir(tmp_path, module_logger):
    generator = make_generator(None)
    run_generation(generator, 1)

    # Nothing written, but the generator still logs to the module logger
    assert os.listdir(tmp_path) == []
    generator._logger.info("still logging")
    assert "still logging" in module_logger.messages


def test_log_file_receives_records_and_closes(tmp_path, module_logger):
    generator = make_generator(tmp_path / "run")
    generator._prepare_output()
    generator._logger.info("after prepare")

    assert "after prepare" in module_logger.messages
    generator.close_log_file()

    with open(os.path.join(generator.output_dir, "log.txt")) as f:
        assert "after prepare" in f.read()
    assert not generator._logger.handlers
