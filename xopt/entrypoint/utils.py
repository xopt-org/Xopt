import logging
import os
import pandas as pd
import sys
import yaml


logger = logging.getLogger(__name__)

_XOPT_INITIAL_COLS = ["xopt_candidate_idx", "xopt_runtime", "xopt_error"]


def normalize_initial_data(df: pd.DataFrame, vocs) -> pd.DataFrame:
    """
    Validate and normalize a user-supplied initial-data DataFrame before
    passing it to Xopt.add_data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from the user's CSV.
    vocs : VOCS
        The VOCS object from the Xopt instance.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with required VOCS columns present, xopt metadata
        columns filled with defaults if absent, and unrecognized columns dropped.
    """
    missing = set(vocs.all_names) - set(df.columns)
    if missing:
        raise ValueError(
            f"Initial data is missing required VOCS columns: {sorted(missing)}"
        )

    df = df.copy()
    if "xopt_candidate_idx" not in df.columns:
        df["xopt_candidate_idx"] = range(len(df))
    if "xopt_runtime" not in df.columns:
        df["xopt_runtime"] = 0.0
    if "xopt_error" not in df.columns:
        df["xopt_error"] = False

    keep = list(vocs.all_names) + _XOPT_INITIAL_COLS
    if "xopt_error_str" in df.columns:
        keep = keep + ["xopt_error_str"]
    extra = sorted(set(df.columns) - set(keep))
    if extra:
        logger.warning(f"Dropping unrecognized columns from initial data: {extra}")
    return df[[c for c in keep if c in df.columns]]


def setup_import_paths(python_path):
    """
    Prepend directories to the module search path.

    Parameters
    ----------
    python_path : list of str
        Directories to add, in priority order. User and environment variables are
        expanded.

    Returns
    -------
    list of str
        Expanded directories that were inserted into sys.path.
    """
    added = []
    for path in python_path:
        expanded = os.path.expanduser(os.path.expandvars(path))
        if expanded not in sys.path:
            sys.path.insert(0, expanded)
            added.append(expanded)

    if added:
        logger.info(f"Python path additions: {added}")

    return added


def override_to_dict(override: str) -> dict:
    """
    Convert strings of form "class_a.class_b.class_c.param=1" to
    {'class_a': {'class_b': {'class_c': {'param': 1}}}}.

    Uses yaml library for consistent type conversion of values following
    same convention as in config files.

    Parameters
    ----------
    override : str
        The override string

    Returns
    -------
    dict
        The nested dictionary containing the value.
    """
    # Check override format
    if "=" not in override:
        raise ValueError(f'Invalid override format: "{override}". Expected key=value')

    path, value = override.split("=", 1)
    yaml_str = (
        "\n".join([" " * idx + x + ":" for idx, x in enumerate(path.split("."))])
        + " "
        + value.strip()
    )
    return yaml.safe_load(yaml_str)


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Nested merging of dicts. Will combine nested dicts keeping keys in both with
    the values in dict2 overriding those in dict1.

    Parameters
    ----------
    dict1 : dict
        Parameters to override
    dict2 : dict
        Parameters used to override those in dict1

    Returns
    -------
    dict
        Values from dict1 with overrides from dict2
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
