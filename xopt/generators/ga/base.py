from pydantic import Field, field_validator
import logging
import os
import pandas as pd
import time

from ..checkpoints import CheckpointMixin
from ..deduplicated import DeduplicatedGeneratorBase

POPULATION_METADATA_COLUMNS = [
    "xopt_generation",
    "xopt_candidate_idx",
    "xopt_runtime",
    "xopt_error",
]


class GAGeneratorBase(CheckpointMixin, DeduplicatedGeneratorBase):
    """
    Base class for genetic algorithm generators which write output and checkpoints.

    Handles the output directory, log file, and periodic checkpointing on behalf of
    subclasses. Subclasses call `end_generation` once each time a generation is
    completed and everything else is taken care of.

    Nothing is written to disk until the generator is used, so building or
    deserializing one never touches the filesystem.

    Parameters
    ----------
    output_dir : str or os.PathLike, optional
        Directory to save algorithm state and population history, or None to write
        nothing. Stored as a string. If the directory already contains data, a number
        is appended to avoid overwriting it.
    checkpoint_freq : int, default=1
        Frequency (in generations) at which checkpoints are saved. Set to -1 to
        disable checkpointing.
    log_level : int
        Level of log messages written to "log.txt".
    """

    output_dir: str | None = None
    checkpoint_freq: int = Field(
        1,
        description="How often (in generations) to save checkpoints (set to -1 to disable)",
    )
    log_level: int = Field(
        logging.INFO, description="Log message level output to log.txt"
    )
    _output_prepared: bool = (
        False  # Whether the output directory has been resolved and created
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value):
        """Accept any os.PathLike, storing it as a string."""
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        return value

    def model_post_init(self, context):
        # Get a unique logger per object. Naming it after the concrete class keeps
        # records propagating through that class's module logger.
        self._logger = logging.getLogger(
            f"{type(self).__module__}.{type(self).__name__}.{id(self)}"
        )
        self._logger.setLevel(self.log_level)

    def _prepare_output(self) -> None:
        """
        Resolve and create the output directory and begin logging to file.

        Repeated calls do nothing. If the requested directory already holds data, a
        number is appended and `output_dir` is updated to the path actually used.
        """
        if (self.output_dir is None) or self._output_prepared:
            return

        # Check if directory exists and do collision avoidance. Resolve into a local
        # so the field is only assigned once, since assignment revalidates the model.
        requested = self.output_dir
        counter = 2
        output_dir = requested
        while os.path.exists(output_dir) and os.listdir(output_dir):
            output_dir = f"{requested}_{counter}"
            counter += 1
        if output_dir != requested:
            self._logger.info(
                f'detected existing output_dir "{requested}" and corrected '
                f'to "{output_dir}" to avoid overwriting'
            )
        self.output_dir = output_dir

        # We are now setup
        os.makedirs(self.output_dir, exist_ok=True)
        self._output_prepared = True

        # Set up file logging
        log_file_path = os.path.join(self.output_dir, "log.txt")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(file_handler)
        self._logger.info(f"routing log output to file: {log_file_path}")

    def end_generation(self, generation_index: int, population: list[dict]) -> None:
        """
        Record a completed generation, writing output and checkpoints as configured.

        Parameters
        ----------
        generation_index : int
            Index of the generation which was just completed.
        population : list of dict
            The individuals making up the completed population.
        """
        self._prepare_output()
        if self.output_dir is None:
            return
        save_start_t = time.perf_counter()

        # Save all Xopt data
        self.data.to_csv(os.path.join(self.output_dir, "data.csv"), index=False)

        # Construct the DataFrame for this population
        pop_df = pd.DataFrame(population)
        pop_df["xopt_generation"] = generation_index

        # Normalize the columns in the DataFrame
        # Avoid schema changing part way through optimization so we can write CSV in append mode
        pop_df = pop_df.reindex(
            columns=self.vocs.all_names + POPULATION_METADATA_COLUMNS
        )

        # Write population DataFrame to file
        csv_path = os.path.join(self.output_dir, "populations.csv")
        pop_df.to_csv(
            csv_path, index=False, mode="a", header=not os.path.isfile(csv_path)
        )
        self._logger.debug(
            f'saved optimization data to "{self.output_dir}" '
            f"in {1000 * (time.perf_counter() - save_start_t):.2f}ms"
        )

        # Save a checkpoint if one is due
        if self.checkpoint_freq > 0 and (generation_index % self.checkpoint_freq == 0):
            checkpoint_path = self._save_checkpoint(
                os.path.join(self.output_dir, "checkpoints")
            )
            self._logger.debug(f'saved checkpoint file "{checkpoint_path}"')

    def close_log_file(self):
        """
        Closes out the log file (if used)
        """
        for handler in list(self._logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self._logger.removeHandler(handler)
