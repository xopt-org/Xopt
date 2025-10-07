import zipfile
from pathlib import Path


def on_pre_build(config):
    """Generate zip files before the build starts."""
    # Configuration for zip files to create
    zip_configs = [
        {
            "source_dir": "docs/examples/ga/nsga2/yaml_interface/assets/yaml_runner_example",
            "output_path": "docs/examples/ga/nsga2/yaml_interface/assets/yaml_runner_example.zip",
        }
    ]

    for zip_config in zip_configs:
        source_dir = Path(zip_config["source_dir"])
        output_path = Path(zip_config["output_path"])

        # Check if source directory exists
        if not source_dir.exists() or not source_dir.is_dir():
            print(
                f"Warning: Source directory {source_dir} does not exist, skipping zip creation"
            )
            continue

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create zip file
        create_zip(source_dir, output_path)
        print(f"Created zip file: {output_path}")


def create_zip(source_dir, output_path):
    """Create a zip file from the source directory."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                # Calculate the archive name (relative path within the zip)
                archive_name = file_path.relative_to(source_dir)
                zipf.write(file_path, archive_name)
