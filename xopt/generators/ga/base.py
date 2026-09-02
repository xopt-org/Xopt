from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
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


class _InstanceLogger(logging.Logger):
    """
    Logger owned by a single object. Used for per-object logging in generator.
    """

    def __reduce__(self):
        return _make_instance_logger, (self.name, self.parent.name, self.level)


def _make_instance_logger(name: str, parent_name: str, level: int) -> logging.Logger:
    """
    Build an unregistered logger which propagates to an existing named logger.

    Parameters
    ----------
    name : str
        Name recorded on emitted records.
    parent_name : str
        Name of the logger records propagate to.
    level : int
        Level of the new logger.

    Returns
    -------
    logging.Logger
    """
    logger = _InstanceLogger(name, level)
    logger.parent = logging.getLogger(parent_name)
    return logger


class _ResolvedOutputDirMixin(BaseModel):
    """
    This mixin just adds a validator to strip the parameter `output_dir_resolved` from input.
    This is saved as a computed field by the  model, but will error on loading from checkpoint
    due to extra fields being disallows. It has to be a mixin here to get run before the
    `CheckpointMixin` is fired.
    """

    @model_validator(mode="before")
    @classmethod
    def drop_resolved_output_dir(cls, values):
        """Discard "output_dir_resolved" so a reloaded generator resolves its own."""
        if isinstance(values, dict) and "output_dir_resolved" in values:
            values = dict(values)
            values.pop("output_dir_resolved")
        return values


class GAGeneratorBase(
    CheckpointMixin, _ResolvedOutputDirMixin, DeduplicatedGeneratorBase
):
    """
    Base class for genetic algorithm generators which write output and checkpoints.

    Handles the output directory, log file, and periodic checkpointing on behalf of
    subclasses. Subclasses call `end_generation` once each time a generation is
    completed and everything else is taken care of.

    Nothing is written to disk until the generator is used, so building or
    deserializing one never touches the filesystem. Each generator owns its logger.

    Parameters
    ----------
    output_dir : str or os.PathLike, optional
        Directory to save algorithm state and population history, or None to write
        nothing. Stored as a string and never modified; environment variables and "~"
        are expanded when the path is used. If the directory already contains data, a
        number is appended to avoid overwriting it.
    checkpoint_freq : int, default=1
        Frequency (in generations) at which checkpoints are saved. Set to -1 to
        disable checkpointing.
    log_level : int
        Level of log messages written to "log.txt".

    Attributes
    ----------
    output_dir_resolved : str
        Expanded, collision free path `output_dir` was resolved to. All file writes go
        here. Empty until the directory has been created.
    """

    output_dir: str | None = Field(
        None,
        description="Directory to save algorithm state and population history, or None "
        "to write nothing. Environment variables and a leading '~' are expanded when "
        "it is used and a number is appended if it already contains data",
    )
    checkpoint_freq: int = Field(
        1,
        description="How often (in generations) to save checkpoints (set to -1 to disable)",
    )
    log_level: int = Field(
        logging.INFO, description="Log message level output to log.txt"
    )
    _output_dir_resolved: str = ""  # Empty until the output directory has been created

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value):
        """Accept any os.PathLike, storing it as a string."""
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        return value

    @computed_field
    @property
    def output_dir_resolved(self) -> str:
        """Directory output is written to, empty until it has been created."""
        return self._output_dir_resolved

    def model_post_init(self, context):
        # Get a unique logger owned by this instance.
        self._logger = _make_instance_logger(
            f"{type(self).__module__}.{type(self).__name__}",
            type(self).__module__,
            self.log_level,
        )

    def _prepare_output(self) -> None:
        """
        Resolve and create the output directory and begin logging to file.

        Repeated calls do nothing. The requested path is expanded and, if it already
        holds data, a number is appended. The result is kept in `output_dir_resolved`.
        """
        if (self.output_dir is None) or self._output_dir_resolved:
            return

        # Check if directory exists and do collision avoidance. Suffixes are applied to
        # the expanded path so that they cannot land inside an environment variable or
        # "~". The path is normalized first so that a trailing separator does not put
        # the suffixed directory inside the one being protected.
        requested = os.path.normpath(
            os.path.expanduser(os.path.expandvars(self.output_dir))
        )
        resolved = requested
        counter = 2
        while os.path.exists(resolved) and os.listdir(resolved):
            resolved = f"{requested}_{counter}"
            counter += 1
        if resolved != requested:
            self._logger.info(
                f'detected existing output_dir "{requested}" and corrected '
                f'to "{resolved}" to avoid overwriting'
            )

        # We are now setup
        os.makedirs(resolved, exist_ok=True)
        self._output_dir_resolved = resolved

        # Set up file logging
        log_file_path = os.path.join(resolved, "log.txt")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(file_handler)
        self._logger.info(f"routing log output to file: {log_file_path}")

        # Record the problem definition alongside the data
        # Note: this is necessary to include in output for users running analysis on the results
        # ie to plot Pareto front, you need to know the names and direction of the objectives
        with open(os.path.join(resolved, "vocs.txt"), "w") as f:
            f.write(self.vocs.model_dump_json())

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
        if not self._output_dir_resolved:
            return
        output_dir = self._output_dir_resolved
        save_start_t = time.perf_counter()

        # Save all Xopt data
        self.data.to_csv(os.path.join(output_dir, "data.csv"), index=False)

        # Construct the DataFrame for this population
        pop_df = pd.DataFrame(population)
        pop_df["xopt_generation"] = generation_index

        # Normalize the columns in the DataFrame
        # Avoid schema changing part way through optimization so we can write CSV in append mode
        pop_df = pop_df.reindex(
            columns=self.vocs.all_names + POPULATION_METADATA_COLUMNS
        )

        # Write population DataFrame to file
        csv_path = os.path.join(output_dir, "populations.csv")
        pop_df.to_csv(
            csv_path, index=False, mode="a", header=not os.path.isfile(csv_path)
        )
        self._logger.debug(
            f'saved optimization data to "{output_dir}" '
            f"in {1000 * (time.perf_counter() - save_start_t):.2f}ms"
        )

        # Save a checkpoint if one is due
        if self.checkpoint_freq > 0 and (generation_index % self.checkpoint_freq == 0):
            checkpoint_path = self._save_checkpoint(
                os.path.join(output_dir, "checkpoints")
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
