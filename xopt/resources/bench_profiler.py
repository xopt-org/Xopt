import os
import sys
import subprocess


def run():
    current_folder = sys.path[0]
    runner_path = current_folder + os.sep + "bench_runner.py"
    benchmark_name = sys.argv[1]
    if "-device" in sys.argv:
        device_index = sys.argv.index("-device") + 1
        device_name = sys.argv[device_index]
        device_name = device_name.replace(":", "_")
        benchmark_name += f"_{device_name}"
    py_spy_command = [
        "py-spy",
        "record",
        "--output",
        f"profile_{benchmark_name}.svg",
        "--format",
        "speedscope",
        # "flamegraph",
        "-r",
        "200",
        # "-n",
        "--",
        "python",
        runner_path,
    ]
    py_spy_command.extend(sys.argv[1:])
    try:
        print("Running command:", " ".join(py_spy_command))
        subprocess.run(py_spy_command)
        print(f"Profile saved to profile_{benchmark_name}.svg")
    except subprocess.CalledProcessError as e:
        print(f"Error running py-spy: {e}")
        print(
            "You might need to run py-spy with sudo, especially on OSX or when attaching to an existing process on Linux."
        )
    except FileNotFoundError:
        print(
            "Error: py-spy command not found. Please ensure py-spy is installed and in your PATH."
        )


if __name__ == "__main__":
    run()
