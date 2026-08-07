from datetime import datetime
from pydantic import BaseModel, Field, model_validator
import json
import os

from ..vocs import VOCS


class CheckpointMixin(BaseModel):
    """
    Mix-in class adding checkpoint saving and loading to a generator.

    Checkpoints are written to a "checkpoints" subdirectory of a caller-supplied
    directory, with the VOCS object written alongside it as "vocs.txt".

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
        # Load the VOCS object
        vocs_fname = os.path.join(os.path.dirname(fname), "../vocs.txt")
        if not os.path.exists(vocs_fname):
            raise ValueError(
                f'Could not load VOCS file at "{vocs_fname}". Complete generator '
                "output directory is required for loading from checkpoint."
            )

        with open(vocs_fname) as f:
            vocs = VOCS(**json.load(f))

        # Load the checkpoint
        with open(fname) as f:
            checkpoint_data = json.load(f)

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
        Write the VOCS object and a checkpoint of the generator state to disk.

        Parameters
        ----------
        path : str or os.PathLike
            Directory into which "vocs.txt" and the "checkpoints" subdirectory
            containing the checkpoint file are written.

        Returns
        -------
        str
            Path to the checkpoint file which was written.
        """
        # Set up the output directory and write the VOCS object needed to reload
        checkpoint_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(path, "vocs.txt"), "w") as f:
            json.dump(self.vocs.model_dump(), f)

        # Create a base filename
        base_checkpoint_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{base_checkpoint_filename}_1.txt"
        )

        # Check if file exists and increment counter until we find a free filename
        counter = 2
        while os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{base_checkpoint_filename}_{counter}.txt"
            )
            counter += 1

        # Now we have a unique filename
        with open(checkpoint_path, "w") as f:
            f.write(self.to_json())

        return checkpoint_path
