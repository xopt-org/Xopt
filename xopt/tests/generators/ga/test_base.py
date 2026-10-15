import json
import logging
import os

import pandas as pd
import pytest

from xopt.generators.ga.base import GAGeneratorBase
from xopt.resources.test_functions.tnk import tnk_vocs
from xopt.vocs import VOCS


class OutputTestGenerator(GAGeneratorBase):
    """Minimal concrete generator for exercising the base class output behavior."""

    name = "ga_base_test"
    supports_single_objective: bool = True
    supports_multi_objective: bool = True
    supports_constraints: bool = True

    def _generate(self, n_candidates: int) -> list[dict]:
        return []


class FsPath:
    """Minimal os.PathLike which is not a pathlib.Path."""

    def __init__(self, path):
        self.path = str(path)

    def __fspath__(self):
        return self.path


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
        output_dir=output_dir,
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


def test_pathlike_output_dir_is_stored_as_a_string(tmp_path):
    requested = tmp_path / "run"

    for value in (requested, FsPath(requested), str(requested)):
        generator = make_generator(value)
        assert isinstance(generator.output_dir, str)
        assert generator.output_dir == str(requested)

        # A path object must not survive into the serialized form
        assert json.loads(generator.to_json())["output_dir"] == str(requested)


def test_environment_variables_are_expanded_but_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("XOPT_TEST_OUTPUT_ROOT", str(tmp_path))
    requested = os.path.join("$XOPT_TEST_OUTPUT_ROOT", "run")
    generator = make_generator(requested)

    run_generation(generator, 1, n_data=4)

    assert generator.output_dir_resolved == str(tmp_path / "run")

    # Storing the unexpanded string is what lets a checkpoint resolve against the
    # environment of whatever machine it is reloaded on
    assert generator.output_dir == requested
    assert json.loads(generator.to_json())["output_dir"] == requested
    assert os.path.isfile(tmp_path / "run" / "data.csv")
    generator.close_log_file()


def test_home_directory_is_expanded_but_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    requested = os.path.join("~", "run")
    generator = make_generator(requested)

    generator._prepare_output()

    assert generator.output_dir == requested
    assert generator.output_dir_resolved == str(tmp_path / "run")
    assert os.path.isdir(tmp_path / "run")
    generator.close_log_file()


def test_vocs_is_written_alongside_the_data(tmp_path):
    # Analysis of the output needs the objective names and directions
    generator = make_generator(tmp_path / "run")
    generator._prepare_output()

    with open(os.path.join(generator.output_dir_resolved, "vocs.txt")) as f:
        written = VOCS(**json.load(f))

    assert written == tnk_vocs
    generator.close_log_file()


def test_prepare_output_creates_directory_and_is_idempotent(tmp_path):
    requested = tmp_path / "run"
    generator = make_generator(requested)

    generator._prepare_output()
    assert generator.output_dir == str(requested)
    assert generator.output_dir_resolved == str(requested)
    assert os.path.isdir(requested)

    generator._prepare_output()
    assert generator.output_dir_resolved == str(requested)
    generator.close_log_file()


def test_existing_empty_directory_is_not_renamed(tmp_path, module_logger):
    # A directory which exists but holds nothing is reused as-is. Tests which hand
    # the generator a TemporaryDirectory depend on this.
    requested = tmp_path / "run"
    os.makedirs(requested)

    generator = make_generator(requested)
    generator._prepare_output()

    assert generator.output_dir == str(requested)
    assert generator.output_dir_resolved == str(requested)
    assert not any("corrected" in m for m in module_logger.messages)
    generator.close_log_file()


def test_non_empty_directory_is_renamed(tmp_path, module_logger):
    requested = tmp_path / "run"
    os.makedirs(requested)
    (requested / "data.csv").write_text("existing\n")

    first = make_generator(requested)
    first._prepare_output()
    assert first.output_dir == str(requested)
    assert first.output_dir_resolved == f"{requested}_2"
    assert os.path.isdir(f"{requested}_2")
    assert any("corrected" in m for m in module_logger.messages)

    # The original directory is left untouched
    assert (requested / "data.csv").read_text() == "existing\n"
    first.close_log_file()

    # A second collision steps to the next suffix
    (tmp_path / "run_2" / "data.csv").write_text("existing\n")
    second = make_generator(requested)
    second._prepare_output()
    assert second.output_dir_resolved == f"{requested}_3"
    second.close_log_file()


def test_trailing_separator_suffix_lands_beside_the_directory(tmp_path):
    requested = tmp_path / "run"
    os.makedirs(requested)
    (requested / "data.csv").write_text("existing\n")

    generator = make_generator(f"{requested}{os.sep}")
    generator._prepare_output()

    assert generator.output_dir_resolved == f"{requested}_2"
    assert os.path.isdir(f"{requested}_2")
    assert not os.path.exists(requested / "_2")
    assert (requested / "data.csv").read_text() == "existing\n"
    generator.close_log_file()


def test_trailing_separator_is_left_alone_without_a_collision(tmp_path):
    requested = f"{tmp_path / 'run'}{os.sep}"
    generator = make_generator(requested)
    generator._prepare_output()

    assert generator.output_dir == requested
    assert generator.output_dir_resolved == str(tmp_path / "run")
    assert os.path.isdir(tmp_path / "run")
    generator.close_log_file()


def test_environment_variable_collision_suffixes_the_expanded_path(
    tmp_path, monkeypatch
):
    # Suffixing the unexpanded string would give "$XOPT_TEST_OUTPUT_ROOT_2", a different
    # and unset variable, leaving a literal directory of that name in the cwd
    root = tmp_path / "root"
    os.makedirs(root)
    (root / "data.csv").write_text("existing\n")
    monkeypatch.setenv("XOPT_TEST_OUTPUT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)

    generator = make_generator("$XOPT_TEST_OUTPUT_ROOT")
    generator._prepare_output()

    assert generator.output_dir == "$XOPT_TEST_OUTPUT_ROOT"
    assert generator.output_dir_resolved == f"{root}_2"
    assert os.path.isdir(f"{root}_2")
    assert (root / "data.csv").read_text() == "existing\n"

    # Nothing named after the variable itself was created
    assert not os.path.exists(tmp_path / "$XOPT_TEST_OUTPUT_ROOT_2")
    generator.close_log_file()


def test_home_directory_collision_suffixes_the_expanded_path(tmp_path, monkeypatch):
    # Likewise "~" must not become a literal "~_2" directory
    home = tmp_path / "home"
    os.makedirs(home)
    (home / "data.csv").write_text("existing\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)

    generator = make_generator("~")
    generator._prepare_output()

    assert generator.output_dir == "~"
    assert generator.output_dir_resolved == f"{home}_2"
    assert os.path.isdir(f"{home}_2")
    assert not os.path.exists(tmp_path / "~_2")
    generator.close_log_file()


def test_resolved_directory_is_reported_but_never_an_input(tmp_path):
    # The generator writes it rather than being given it, so a reloaded generator must
    # resolve one of its own instead of appending to the earlier run
    requested = tmp_path / "run"
    generator = make_generator(requested)

    assert generator.output_dir_resolved == ""
    assert json.loads(generator.to_json())["output_dir_resolved"] == ""

    generator._prepare_output()
    dumped = json.loads(generator.to_json())
    assert dumped["output_dir_resolved"] == str(requested)

    reloaded = OutputTestGenerator.model_validate(dumped)
    assert reloaded.output_dir == str(requested)
    assert reloaded.output_dir_resolved == ""

    reloaded._prepare_output()
    assert reloaded.output_dir_resolved == f"{requested}_2"
    generator.close_log_file()
    reloaded.close_log_file()


def test_checkpoint_resume_resolves_a_fresh_directory(tmp_path):
    requested = tmp_path / "run"
    generator = make_generator(requested)
    run_generation(generator, 1)
    generator.close_log_file()

    checkpoint_dir = os.path.join(generator.output_dir_resolved, "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])

    restored = OutputTestGenerator(checkpoint_file=checkpoint_file)
    assert restored.output_dir == str(requested)
    assert restored.output_dir_resolved == ""

    restored._prepare_output()
    assert restored.output_dir_resolved == f"{requested}_2"
    restored.close_log_file()

    # A directory given alongside the checkpoint still wins
    override = OutputTestGenerator(
        checkpoint_file=checkpoint_file, output_dir=tmp_path / "other"
    )
    assert override.output_dir == str(tmp_path / "other")
    assert override.output_dir_resolved == ""


def test_end_generation_writes_both_files(tmp_path):
    generator = make_generator(tmp_path / "run")
    run_generation(generator, 1, n_data=8)

    assert (
        len(pd.read_csv(os.path.join(generator.output_dir_resolved, "data.csv"))) == 8
    )

    pop_df = pd.read_csv(os.path.join(generator.output_dir_resolved, "populations.csv"))
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
    assert (
        len(pd.read_csv(os.path.join(generator.output_dir_resolved, "data.csv"))) == 8
    )

    # populations.csv is appended and carries exactly one header line
    population_path = os.path.join(generator.output_dir_resolved, "populations.csv")
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

    pop_df = pd.read_csv(os.path.join(generator.output_dir_resolved, "populations.csv"))
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

    checkpoint_dir = os.path.join(generator.output_dir_resolved, "checkpoints")
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

    with open(os.path.join(generator.output_dir_resolved, "log.txt")) as f:
        assert "after prepare" in f.read()
    assert not generator._logger.handlers


def test_loggers_are_not_registered_globally(tmp_path):
    # Naming a logger after id(self) leaks it into the global registry and collides
    # once ids are recycled, so later generators inherit earlier ones' file handlers.
    registered_before = set(logging.Logger.manager.loggerDict)

    # Held in a list so that no id can be recycled part way through
    generators = [make_generator(tmp_path / f"run_{index}") for index in range(20)]

    assert len({id(generator._logger) for generator in generators}) == 20
    assert set(logging.Logger.manager.loggerDict) - registered_before <= {
        OutputTestGenerator.__module__
    }


def test_log_files_are_not_shared_between_generators(tmp_path):
    first = make_generator(tmp_path / "first")
    second = make_generator(tmp_path / "second")
    first._prepare_output()
    second._prepare_output()

    assert len(first._logger.handlers) == 1
    assert len(second._logger.handlers) == 1

    first._logger.info("from the first generator")
    second._logger.info("from the second generator")
    first.close_log_file()
    second.close_log_file()

    with open(os.path.join(first.output_dir_resolved, "log.txt")) as f:
        first_log = f.read()
    with open(os.path.join(second.output_dir_resolved, "log.txt")) as f:
        second_log = f.read()

    assert "from the first generator" in first_log
    assert "from the second generator" not in first_log
    assert "from the second generator" in second_log
    assert "from the first generator" not in second_log
