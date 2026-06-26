import numpy as np
import pandas as pd
import pytest
import sys
import yaml
from unittest import mock
from xopt import entrypoint
from xopt.entrypoint import normalize_initial_data
from xopt.resources.test_functions.tnk import tnk_vocs


def _tnk_df(n=3):
    return pd.DataFrame(
        {
            "x1": np.random.uniform(0, np.pi, n),
            "x2": np.random.uniform(0, np.pi, n),
            "y1": np.random.uniform(0, 1, n),
            "y2": np.random.uniform(0, 1, n),
            "c1": np.random.uniform(-1, 1, n),
            "c2": np.random.uniform(0, 0.5, n),
            "a": [1.0] * n,
        }
    )


class TestEntryPointScript:
    def make_config(self, tmp_path):
        config = {"foo": "bar"}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        return str(config_path), config

    @mock.patch("xopt.entrypoint.remove_none_values", side_effect=lambda x: x)
    @mock.patch("xopt.entrypoint.Xopt")
    def test_main_basic(self, mock_Xopt, mock_remove_none, tmp_path):
        config_path, config = self.make_config(tmp_path)
        sys_argv = [
            "entrypoint.py",
            config_path,
            "--executor",
            "map",
            "--max_workers",
            "2",
            "--override",
            "foo=baz",
            "--verbose",
        ]
        with mock.patch.object(sys, "argv", sys_argv):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=yaml.dump(config))
            ):
                mock_xopt_instance = mock.Mock()
                mock_Xopt.model_validate.return_value = mock_xopt_instance
                mock_xopt_instance.run.return_value = None
                entrypoint.main()
                assert mock_Xopt.model_validate.called
                assert mock_xopt_instance.run.called

    def test_override_to_dict(self):
        s = "a.b.c=42"
        result = entrypoint.override_to_dict(s)
        assert result == {"a": {"b": {"c": 42}}}

    def test_merge_dicts(self):
        d1 = {"a": {"b": 1}, "c": 2}
        d2 = {"a": {"b": 3}, "d": 4}
        merged = entrypoint.merge_dicts(d1, d2)
        assert merged == {"a": {"b": 3}, "c": 2, "d": 4}

    def test_get_executor_map(self):
        with entrypoint.get_executor("map") as exe:
            assert exe.__class__.__name__ == "DummyExecutor"

    def test_get_executor_invalid(self):
        with pytest.raises(ValueError, match="Unknown executor: bad"):
            with entrypoint.get_executor("bad"):
                pass

    def test_normalize_initial_data_missing_cols(self):
        df = _tnk_df().drop(columns=["y2"])
        with pytest.raises(ValueError, match="y2"):
            normalize_initial_data(df, tnk_vocs)

    def test_normalize_initial_data_adds_xopt_defaults(self):
        df = _tnk_df()
        result = normalize_initial_data(df, tnk_vocs)
        assert list(result["xopt_candidate_idx"]) == list(range(len(df)))
        assert (result["xopt_runtime"] == 0.0).all()
        assert (result["xopt_error"] == False).all()  # noqa: E712

    def test_normalize_initial_data_drops_extra_cols(self):
        df = _tnk_df()
        df["extra_col"] = 999
        result = normalize_initial_data(df, tnk_vocs)
        assert "extra_col" not in result.columns

    def test_normalize_initial_data_keeps_xopt_error_str(self):
        df = _tnk_df()
        df["xopt_error_str"] = "some error"
        result = normalize_initial_data(df, tnk_vocs)
        assert "xopt_error_str" in result.columns
