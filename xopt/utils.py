import pandas as pd

from .vocs import VOCS
from xopt.generators import generators, generator_default_options


def get_generator_and_defaults(name: str):
    return generators[name], generator_default_options[name]


def add_constraint_information(data: pd.DataFrame, vocs: VOCS) -> pd.DataFrame:
    """
    determine if constraints have been satisfied based on data and vocs

    """
    temp_data = data.copy()

    # transform data s.t. negative values imply feasibility
    constraints = vocs.constraints

    for name, value in constraints.items():
        if value[0] == "GREATER_THAN":
            temp_data[name] = -(data[name] - value[1])
        else:
            temp_data[name] = data[name] - value[1]

        # add column to original dataframe
        data[f"{name}_feas"] = temp_data[name] < 0.0

    # add a column feas to show feasibility based on all of the constraints
    data["feas"] = data[[f"{ele}_feas" for ele in constraints]].all(axis=1)

    return data
