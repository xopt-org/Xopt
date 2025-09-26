import os
import sys
import subprocess


def run():
    current_folder = sys.path[0]
    runner_path = current_folder + os.sep + 'prunner.py'
    benchmark_name = sys.argv[1]
    py_spy_command = ["py-spy", "record", "--output", f"profile_{benchmark_name}.svg", "--format", "flamegraph", "--", "python",
                      runner_path]
    py_spy_command.extend(sys.argv[1:])
    try:
        print("Running command:", ' '.join(py_spy_command))
        subprocess.run(py_spy_command)
    except subprocess.CalledProcessError as e:
        print(f"Error running py-spy: {e}")
        print("You might need to run py-spy with sudo, especially on OSX or when attaching to an existing process on Linux.")
    except FileNotFoundError:
        print("Error: py-spy command not found. Please ensure py-spy is installed and in your PATH.")


if __name__ == "__main__":
    run()