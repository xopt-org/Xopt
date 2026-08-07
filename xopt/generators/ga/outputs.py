import os
import pandas as pd

from ...vocs import VOCS

POPULATION_METADATA_COLUMNS = [
    "xopt_generation",
    "xopt_candidate_idx",
    "xopt_runtime",
    "xopt_error",
]


class GAOutputs:
    """
    File output for genetic algorithm generators.

    Owns the layout of the output directory and writes the evaluated data and each
    completed population to disk. The directory is resolved and created when this
    object is constructed.
    """

    def __init__(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            Requested directory for output. If it already exists and is not empty, a
            numeric suffix is appended to avoid overwriting previous data. The path
            actually used is available as the `output_dir` attribute.
        """
        self.requested_output_dir = output_dir

        # Check if directory exists and do collision avoidance
        counter = 2
        self.output_dir = output_dir
        while os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            self.output_dir = f"{output_dir}_{counter}"
            counter += 1

        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def data_path(self) -> str:
        """Path of the file holding every evaluated individual."""
        return os.path.join(self.output_dir, "data.csv")

    @property
    def population_path(self) -> str:
        """Path of the file holding each completed population."""
        return os.path.join(self.output_dir, "populations.csv")

    @property
    def checkpoint_dir(self) -> str:
        """Directory into which checkpoint files are written."""
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def log_path(self) -> str:
        """Path of the log file."""
        return os.path.join(self.output_dir, "log.txt")

    def register_generation(
        self,
        generation_index: int,
        population: list[dict],
        data: pd.DataFrame,
        vocs: VOCS,
    ) -> None:
        """
        Write a completed generation to disk.

        Parameters
        ----------
        generation_index : int
            Index recorded in the "xopt_generation" column of the population file.
        population : list of dict
            The individuals making up the completed population.
        data : pd.DataFrame
            All data evaluated so far. Overwrites the data file.
        vocs : VOCS
            Used to normalize the columns of the population file.
        """
        # Save all Xopt data
        data.to_csv(self.data_path, index=False)

        # Construct the DataFrame for this population
        pop_df = pd.DataFrame(population)
        pop_df["xopt_generation"] = generation_index

        # Normalize the columns in the DataFrame
        # Avoid schema changing part way through optimization so we can write CSV in append mode
        pop_df = pop_df.reindex(columns=vocs.all_names + POPULATION_METADATA_COLUMNS)

        # Write population DataFrame to file
        csv_path = self.population_path
        pop_df.to_csv(
            csv_path, index=False, mode="a", header=not os.path.isfile(csv_path)
        )
