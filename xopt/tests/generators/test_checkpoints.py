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


def test_save_checkpoint_layout(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    checkpoint_path = generator._save_checkpoint(tmp_path)

    # VOCS is written to the parent directory and the checkpoint into "checkpoints"
    vocs_path = tmp_path / "vocs.txt"
    assert vocs_path.is_file()
    assert os.path.dirname(checkpoint_path) == str(tmp_path / "checkpoints")
    assert os.listdir(tmp_path / "checkpoints") == [os.path.basename(checkpoint_path)]

    # Filename follows the timestamp plus deduplication index scheme
    _, index = parse_checkpoint_filename(os.path.basename(checkpoint_path))
    assert index == 1

    # Both files hold valid JSON and the VOCS object round trips
    with open(checkpoint_path) as f:
        assert "counter" in json.load(f)
    with open(vocs_path) as f:
        assert VOCS(**json.load(f)) == generator.vocs


def test_checkpoint_round_trip(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE), counter=17)
    checkpoint_path = generator._save_checkpoint(tmp_path)

    # VOCS comes from the checkpoint output directory, not from the user
    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)
    assert reloaded.counter == 17
    assert reloaded.vocs == generator.vocs

    # The path used to load is not carried into the reloaded generator's state
    assert reloaded.checkpoint_file is None
    assert "checkpoint_file" not in json.loads(reloaded.to_json())


def test_checkpoint_user_values_take_precedence(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE), counter=17)
    checkpoint_path = generator._save_checkpoint(tmp_path)

    reloaded = CheckpointingRandomGenerator(checkpoint_file=checkpoint_path, counter=99)
    assert reloaded.counter == 99


def test_save_checkpoint_avoids_overwriting(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    first = generator._save_checkpoint(tmp_path)
    second = generator._save_checkpoint(tmp_path)

    assert first != second
    assert len(os.listdir(tmp_path / "checkpoints")) == 2


def test_load_checkpoint_missing_vocs(tmp_path):
    generator = CheckpointingRandomGenerator(vocs=deepcopy(TEST_VOCS_BASE))
    checkpoint_path = generator._save_checkpoint(tmp_path)
    os.remove(tmp_path / "vocs.txt")

    with pytest.raises(ValueError, match="Could not load VOCS file"):
        CheckpointingRandomGenerator(checkpoint_file=checkpoint_path)
