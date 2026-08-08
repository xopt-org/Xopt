from pydantic import Field, PrivateAttr, computed_field, model_validator
from typing import Any
import logging
import time

from ..checkpoints import CheckpointMixin
from ..deduplicated import DeduplicatedGeneratorBase
from .outputs import GAOutputs


class GAGeneratorBase(CheckpointMixin, DeduplicatedGeneratorBase):
    """
    Base class for genetic algorithm generators which write output and checkpoints.

    Handles the output directory, log file, and periodic checkpointing on behalf of
    subclasses. Subclasses call `end_generation` once each time a generation is
    completed and everything else is taken care of.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save algorithm state and population history, or None to write
        nothing. If the directory already contains data, a number is appended to
        avoid overwriting it.
    checkpoint_freq : int, default=1
        Frequency (in generations) at which checkpoints are saved. Set to -1 to
        disable checkpointing.
    log_level : int
        Level of log messages written to "log.txt".
    """

    checkpoint_freq: int = Field(
        1,
        description="How often (in generations) to save checkpoints (set to -1 to disable)",
    )
    log_level: int = Field(
        logging.INFO, description="Log message level output to log.txt"
    )
    _outputs: GAOutputs | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _build_outputs(cls, data: Any, handler):
        """
        Hand ownership of "output_dir" to the GAOutputs object.

        The checkpoint is merged here rather than being left to the inherited
        "before" validator, which pydantic would run inside this one. Doing it first
        means "output_dir" is taken from the fully merged data, and consuming
        "checkpoint_file" leaves the inherited validator with nothing to do.
        """
        output_dir = None
        if isinstance(data, dict):
            data = cls.load_from_checkpoint(dict(data))
            output_dir = data.pop("output_dir", None)

        instance = handler(data)

        # Assigning to any field re-runs this validator, which must not discard the
        # object already holding the output directory and log file
        if instance._outputs is None:
            instance._outputs = GAOutputs(
                output_dir,
                f"{type(instance).__module__}.{type(instance).__name__}",
                instance.log_level,
            )
            instance._logger = instance._outputs.logger
        return instance

    @computed_field
    @property
    def output_dir(self) -> str | None:
        """Directory output is written to, or None if output is disabled."""
        return self._outputs.output_dir

    @output_dir.setter
    def output_dir(self, value: str | None) -> None:
        self._outputs.output_dir = value

    def get_output(self) -> GAOutputs:
        """
        Returns the object handling file output and logging.

        The output directory is created on the first call, so nothing is written to
        disk until the generator is actually used.
        """
        requested = self._outputs.prepare()
        if requested is not None:
            self._logger.info(
                f'detected existing output_dir "{requested}" and corrected '
                f'to "{self._outputs.output_dir}" to avoid overwriting'
            )
        return self._outputs

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
        output = self.get_output()
        if not output.enabled:
            return

        # Write the evaluated data and this population to disk
        save_start_t = time.perf_counter()
        output.register_generation(generation_index, population, self.data, self.vocs)
        self._logger.debug(
            f'saved optimization data to "{output.output_dir}" '
            f"in {1000 * (time.perf_counter() - save_start_t):.2f}ms"
        )

        # Save a checkpoint if one is due
        if self.checkpoint_freq > 0 and (generation_index % self.checkpoint_freq == 0):
            checkpoint_path = self._save_checkpoint(output.checkpoint_dir)
            self._logger.debug(f'saved checkpoint file "{checkpoint_path}"')

    def close_log_file(self):
        """
        Closes out the log file (if used)
        """
        self._outputs.close_log()
