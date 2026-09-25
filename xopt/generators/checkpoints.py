from datetime import datetime
from pydantic import BaseModel, Field, model_validator
import json
import os

from ..vocs import VOCS


class CheckpointMixin(BaseModel):
    """
    Mix-in class adding checkpoint saving and loading to a generator.

    Checkpoints are written to a caller-supplied directory. The VOCS object is
    serialized into the checkpoint itself. Legacy checkpoints which predate this
    instead carry the VOCS object in a "vocs.txt" file one level above the
    directory holding the checkpoints and are still supported when loading.

    The host class must provide a ``vocs`` attribute and a ``to_json`` method.
    Writing of the checkpoint file is the responsibility of the concrete class
    as it will be implementation dependent.

    Parameters
    ----------
    checkpoint_file : str, optional
        Path to checkpoint file to load from. If provided, the generator will be
        initialized from the checkpoint state. User-specified parameters will
        override checkpoint values.
    """

    checkpoint_file: str | None = Field(
        None, description="Path to checkpoint file to load from", exclude=True
    )

    @staticmethod
    def _load_checkpoint_data(fname: str) -> dict:
        """
        Internal function to load generator data from checkpoint file as well as VOCS object.

        Parameters
        ----------
        fname : str
            Path to the checkpoint file

        Returns
        -------
        dict
            Dictionary containing VOCS and checkpoint data
        """
        # Load the checkpoint
        with open(fname) as f:
            checkpoint_data = json.load(f)

        if "vocs" in checkpoint_data:
            return checkpoint_data

        # Legacy checkpoints w/o VOCS
        vocs_fname = os.path.join(os.path.dirname(fname), "../vocs.txt")
        if not os.path.exists(vocs_fname):
            raise ValueError(
                f'Checkpoint "{fname}" does not contain a VOCS object and no '
                f'VOCS file was found at "{vocs_fname}".'
            )

        with open(vocs_fname) as f:
            vocs = VOCS(**json.load(f))

        return {"vocs": vocs, **checkpoint_data}

    @model_validator(mode="before")
    @classmethod
    def load_from_checkpoint(cls, values):
        """
        Load from checkpoint file if checkpoint_file is provided.
        """
        # Case when a checkpoint file has been supplied
        if isinstance(values, dict) and "checkpoint_file" in values:
            checkpoint_file = values.pop("checkpoint_file")
            if checkpoint_file is not None:
                # Load checkpoint data
                checkpoint_data = cls._load_checkpoint_data(checkpoint_file)

                # Merge with user data precedence
                merged_data = {**checkpoint_data, **values}
                return merged_data

        # No checkpoint
        return values

    def _save_checkpoint(self, path: str | os.PathLike) -> str:
        """
        Write a checkpoint of the generator state to disk.

        Parameters
        ----------
        path : str or os.PathLike
            Directory into which the checkpoint file is written. Created if it
            does not already exist.

        Returns
        -------
        str
            Path to the checkpoint file which was written.
        """
        # Set up the output directory
        os.makedirs(path, exist_ok=True)

        # Create a base filename
        base_checkpoint_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(path, f"{base_checkpoint_filename}_1.txt")

        # Check if file exists and increment counter until we find a free filename
        counter = 2
        while os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(
                path, f"{base_checkpoint_filename}_{counter}.txt"
            )
            counter += 1

        # Now we have a unique filename
        with open(checkpoint_path, "w") as f:
            f.write(self.to_json())

        return checkpoint_path
