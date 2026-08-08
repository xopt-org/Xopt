import logging
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
    File output and logging for genetic algorithm generators.

    Owns the output directory and the logger used by its owner. Construction is
    cheap: the directory is not resolved or created until `prepare` is called, so
    merely building or deserializing a generator never touches the filesystem.
    """

    def __init__(
        self,
        output_dir: str | None,
        logger_name: str,
        log_level: int = logging.INFO,
    ):
        """
        Parameters
        ----------
        output_dir : str, optional
            Directory for output, or None to disable file output. If it already
            exists and is not empty, a numeric suffix is appended on `prepare` to
            avoid overwriting previous data.
        logger_name : str
            Name the logger is created beneath. Records propagate to this logger,
            so it should name the owner's module for log configuration to work as
            users expect.
        log_level : int
            Level applied to the logger and to the log file.
        """
        self.output_dir = output_dir
        self.log_level = log_level
        self._prepared = False

        self._logger = logging.getLogger(f"{logger_name}.{id(self)}")
        self._logger.setLevel(log_level)

    def __setattr__(self, name, value):
        # Pointing at a new directory requires setting that directory up again
        if name == "output_dir" and getattr(self, "_prepared", False):
            self._prepared = False
        super().__setattr__(name, value)

    @property
    def logger(self) -> logging.Logger:
        """Logger for the owner to write to. Also writes to the log file once prepared."""
        return self._logger

    @property
    def enabled(self) -> bool:
        """Whether output is written to disk at all."""
        return self.output_dir is not None

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

    def prepare(self) -> str | None:
        """
        Resolve and create the output directory and begin logging to file.

        Repeated calls do nothing. If the requested directory already holds data, a
        numeric suffix is appended and `output_dir` is updated to the path used.

        Returns
        -------
        str or None
            The directory which was requested, if it differed from the one used, and
            None otherwise. Lets the caller report the correction.
        """
        if self._prepared or not self.enabled:
            return None

        # Check if directory exists and do collision avoidance
        requested = self.output_dir
        counter = 2
        while os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            self.output_dir = f"{requested}_{counter}"
            counter += 1

        os.makedirs(self.output_dir, exist_ok=True)
        self._prepared = True

        # Route log output to a file inside the output directory
        file_handler = logging.FileHandler(self.log_path, mode="w")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(file_handler)
        self._logger.info(f"routing log output to file: {self.log_path}")

        return requested if self.output_dir != requested else None

    def close_log(self):
        """
        Close out the log file, if one was opened.
        """
        for handler in list(self._logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self._logger.removeHandler(handler)

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
        if not self.enabled:
            return
        self.prepare()

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
