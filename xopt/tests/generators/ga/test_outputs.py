import logging
import os

import pandas as pd
import pytest

from xopt.generators.ga.outputs import GAOutputs
from xopt.resources.test_functions.tnk import tnk_vocs

LOGGER_NAME = "xopt.tests.generators.ga.test_outputs"


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


def test_construction_touches_nothing(tmp_path):
    # Building the object must not create anything. Generators are constructed and
    # deserialized freely, and each round trip would otherwise leave a directory.
    requested = str(tmp_path / "run")
    outputs = GAOutputs(requested, LOGGER_NAME)

    assert outputs.output_dir == requested
    assert outputs.enabled
    assert os.listdir(tmp_path) == []


def test_prepare_creates_directory_and_is_idempotent(tmp_path):
    requested = str(tmp_path / "run")
    outputs = GAOutputs(requested, LOGGER_NAME)

    assert outputs.prepare() is None
    assert outputs.output_dir == requested
    assert os.path.isdir(requested)

    # A second call changes nothing
    assert outputs.prepare() is None
    assert outputs.output_dir == requested


def test_existing_empty_directory_is_not_renamed(tmp_path):
    # A directory which exists but holds nothing is reused as-is. Tests which hand
    # the generator a TemporaryDirectory depend on this.
    requested = str(tmp_path / "run")
    os.makedirs(requested)

    outputs = GAOutputs(requested, LOGGER_NAME)
    assert outputs.prepare() is None
    assert outputs.output_dir == requested


def test_non_empty_directory_is_renamed(tmp_path):
    requested = str(tmp_path / "run")
    os.makedirs(requested)
    (tmp_path / "run" / "data.csv").write_text("existing\n")

    first = GAOutputs(requested, LOGGER_NAME)
    assert first.prepare() == requested
    assert first.output_dir == f"{requested}_2"
    assert os.path.isdir(f"{requested}_2")

    # The original directory is left untouched
    assert (tmp_path / "run" / "data.csv").read_text() == "existing\n"

    # A second collision steps to the next suffix
    (tmp_path / "run_2" / "data.csv").write_text("existing\n")
    second = GAOutputs(requested, LOGGER_NAME)
    second.prepare()
    assert second.output_dir == f"{requested}_3"


def test_assigning_output_dir_reprepares(tmp_path):
    outputs = GAOutputs(str(tmp_path / "first"), LOGGER_NAME)
    outputs.prepare()

    outputs.output_dir = str(tmp_path / "second")
    assert outputs.prepare() is None
    assert os.path.isdir(tmp_path / "second")


def test_disabled_outputs(tmp_path):
    outputs = GAOutputs(None, LOGGER_NAME)

    assert not outputs.enabled
    assert outputs.prepare() is None

    # Registering a generation is a no-op rather than an error
    outputs.register_generation(1, make_population(4, 0), make_data(4), tnk_vocs)
    assert os.listdir(tmp_path) == []


def test_paths_resolve_under_resolved_directory(tmp_path):
    requested = str(tmp_path / "run")
    os.makedirs(requested)
    (tmp_path / "run" / "data.csv").write_text("existing\n")

    outputs = GAOutputs(requested, LOGGER_NAME)
    outputs.prepare()
    resolved = f"{requested}_2"
    assert outputs.data_path == os.path.join(resolved, "data.csv")
    assert outputs.population_path == os.path.join(resolved, "populations.csv")
    assert outputs.checkpoint_dir == os.path.join(resolved, "checkpoints")
    assert outputs.log_path == os.path.join(resolved, "log.txt")


def test_register_generation_writes_both_files(tmp_path):
    outputs = GAOutputs(str(tmp_path / "run"), LOGGER_NAME)
    outputs.register_generation(1, make_population(4, 0), make_data(8), tnk_vocs)

    data_df = pd.read_csv(outputs.data_path)
    assert len(data_df) == 8

    pop_df = pd.read_csv(outputs.population_path)
    assert len(pop_df) == 4
    assert (pop_df["xopt_generation"] == 1).all()
    assert list(pop_df.columns) == tnk_vocs.all_names + [
        "xopt_generation",
        "xopt_candidate_idx",
        "xopt_runtime",
        "xopt_error",
    ]


def test_data_overwritten_while_populations_accumulate(tmp_path):
    outputs = GAOutputs(str(tmp_path / "run"), LOGGER_NAME)
    outputs.register_generation(1, make_population(4, 0), make_data(4), tnk_vocs)
    outputs.register_generation(2, make_population(4, 1), make_data(8), tnk_vocs)

    # data.csv is a full overwrite, so it reflects only the latest call
    assert len(pd.read_csv(outputs.data_path)) == 8

    # populations.csv is appended and carries exactly one header line
    pop_df = pd.read_csv(outputs.population_path)
    assert len(pop_df) == 8
    assert sorted(pop_df["xopt_generation"].unique()) == [1, 2]
    with open(outputs.population_path) as f:
        header_count = sum(1 for line in f if line.startswith("x1,"))
    assert header_count == 1


def test_register_generation_normalizes_changing_schema(tmp_path):
    outputs = GAOutputs(str(tmp_path / "run"), LOGGER_NAME)
    outputs.register_generation(1, make_population(4, 0), make_data(4), tnk_vocs)

    # A later generation gaining an extra key must not shift the appended columns
    outputs.register_generation(
        2, make_population(4, 1, extra={"obs1": 3.0}), make_data(8), tnk_vocs
    )

    sparse = make_population(4, 2)
    for individual in sparse:
        del individual["xopt_runtime"]
    outputs.register_generation(3, sparse, make_data(12), tnk_vocs)

    pop_df = pd.read_csv(outputs.population_path)
    assert len(pop_df) == 12
    assert "obs1" not in pop_df.columns
    assert pop_df[pop_df["xopt_generation"] == 3]["xopt_runtime"].isna().all()
    assert pop_df[pop_df["xopt_generation"] == 2]["xopt_runtime"].notna().all()


class RecordingHandler(logging.Handler):
    """Captures messages so propagation to the parent logger can be checked."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def parent_logger():
    """The logger GAOutputs loggers are created beneath, with a capturing handler."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = RecordingHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)


def test_logger_propagates_without_output(parent_logger):
    outputs = GAOutputs(None, LOGGER_NAME, log_level=logging.DEBUG)

    outputs.logger.info("no output configured")
    assert "no output configured" in parent_logger.messages


def test_logger_writes_to_file_and_propagates(tmp_path, parent_logger):
    outputs = GAOutputs(str(tmp_path / "run"), LOGGER_NAME, log_level=logging.DEBUG)
    outputs.prepare()
    outputs.logger.info("after prepare")
    outputs.close_log()

    # Reaches the parent logger
    assert "after prepare" in parent_logger.messages

    # ... and the log file
    with open(outputs.log_path) as f:
        assert "after prepare" in f.read()


def test_close_log_removes_handlers(tmp_path):
    outputs = GAOutputs(str(tmp_path / "run"), LOGGER_NAME)
    outputs.prepare()
    assert outputs.logger.handlers

    outputs.close_log()
    assert not outputs.logger.handlers
