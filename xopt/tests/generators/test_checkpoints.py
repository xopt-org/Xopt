from copy import deepcopy
from datetime import datetime
import json
import os

import pytest

from xopt.generators.checkpoints import CheckpointMixin
from xopt.generators.random import RandomGenerator
from xopt.resources.testing import TEST_VOCS_BASE
from xopt.vocs import VOCS


class CheckpointingRandomGenerator(CheckpointMixin, RandomGenerator):
    """Minimal host for the checkpoint mixin with a field to round trip."""

    counter: int = 0


def parse_checkpoint_filename(filename: str) -> tuple[datetime, int]:
    """Split a checkpoint filename into its timestamp and deduplication index."""
    base, index = filename.rsplit("_", 1)
    return datetime.strptime(base, "%Y%m%d_%H%M%S"), int(index.split(".")[0])


def make_legacy_checkpoint(checkpoint_path: str) -> None:
    """Rewrite a checkpoint into the legacy layout with VOCS held in "vocs.txt"."""
    with open(checkpoint_path) as f:
        checkpoint_data = json.load(f)
    vocs = checkpoint_data.pop("vocs")

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)
    legacy_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    with open(os.path.join(legacy_dir, "vocs.txt"), "w") as f:
        json.dump(vocs, f)


def test_save_checkpoint_layout(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")

    # The checkpoint is written directly into the supplied directory
    assert os.path.dirname(checkpoint_path) == str(tmp_path / "checkpoints")
    assert os.listdir(tmp_path / "checkpoints") == [os.path.basename(checkpoint_path)]
    assert os.listdir(tmp_path) == ["checkpoints"]

    # Filename follows the timestamp plus deduplication index scheme
    _, index = parse_checkpoint_filename(os.path.basename(checkpoint_path))
    assert index == 1

    # The checkpoint holds valid JSON and carries the VOCS object itself
    with open(checkpoint_path) as f:
        checkpoint_data = json.load(f)
    assert "counter" in checkpoint_data
    assert VOCS(**checkpoint_data["vocs"]) == generator.vocs


def test_checkpoint_round_trip(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE), counter=17)
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")

    # VOCS comes from the checkpoint itself, not from the user
    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)
    assert reloaded.counter == 17
    assert reloaded.vocs == generator.vocs

    # The path used to load is not carried into the reloaded generator's state
    assert reloaded.checkpoint_file is None
    assert "checkpoint_file" not in json.loads(reloaded.to_json())


def test_checkpoint_user_values_take_precedence(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE), counter=17)
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")

    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path, counter=99)
    assert reloaded.counter == 99


def test_save_checkpoint_avoids_overwriting(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    first = generator._save_checkpoint(tmp_path / "checkpoints")
    second = generator._save_checkpoint(tmp_path / "checkpoints")

    assert first != second
    assert len(os.listdir(tmp_path / "checkpoints")) == 2


def test_load_legacy_checkpoint(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE), counter=17)
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")
    make_legacy_checkpoint(checkpoint_path)

    # VOCS is recovered from "vocs.txt" since the checkpoint does not carry it
    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)
    assert reloaded.counter == 17
    assert reloaded.vocs == generator.vocs


def test_load_legacy_checkpoint_missing_vocs_file(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")
    make_legacy_checkpoint(checkpoint_path)
    os.remove(tmp_path / "vocs.txt")

    with pytest.raises(ValueError, match="does not contain a VOCS object"):
        CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)


def test_embedded_vocs_preferred_over_legacy_file(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    checkpoint_path = generator._save_checkpoint(tmp_path / "checkpoints")

    # A stale legacy file beside a modern checkpoint must be ignored
    stale_vocs = deepcopy(TEST_VOCS_BASE)
    stale_vocs.variables.pop(next(iter(stale_vocs.variables)))
    with open(tmp_path / "vocs.txt", "w") as f:
        json.dump(stale_vocs.model_dump(), f)

    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)
    assert reloaded.vocs == generator.vocs
