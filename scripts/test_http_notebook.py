#!/usr/bin/env python3
"""Execute an Xopt HTTP notebook with an isolated local backend."""

import argparse
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


NOTEBOOKS = {
    "http_generator_ucb.ipynb": "xopt.generators.remote.server:app",
    "http_generator_bax.ipynb": "xopt.generators.remote.server:app",
    "http_generator_custom_model.ipynb": (
        "xopt.generators.remote.cloudpickle_server:app"
    ),
}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(url, process, timeout=30):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"HTTP server exited with status {process.returncode}")

        try:
            with urllib.request.urlopen(f"{url}/openapi.json", timeout=1):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)

    raise TimeoutError(f"HTTP server did not start: {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument(
        "--runner",
        choices=("python", "uv"),
        default="python",
    )
    args = parser.parse_args()

    notebook = Path(args.notebook)
    backend = NOTEBOOKS.get(notebook.name)

    if backend is None:
        parser.error(f"Not an HTTP notebook: {notebook}")

    port = free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["SMOKE_TEST"] = "True"
    env.pop("XOPT_HTTP_GENERATOR_TOKEN", None)
    env["XOPT_HTTP_URL"] = url
    env["XOPT_REMOTE_URL"] = url

    with tempfile.TemporaryDirectory(prefix="xopt-notebook-") as tmp:
        if notebook.name == "http_generator_custom_model.ipynb":
            token = secrets.token_urlsafe(32)
            token_file = Path(tmp) / "token"
            token_file.write_text(token)
            token_file.chmod(0o600)

            env["XOPT_HTTP_GENERATOR_TOKEN"] = token
            env["XOPT_TOKEN_FILE"] = str(token_file)

        if args.runner == "uv":
            server_command = [
                "uv",
                "run",
                "--no-sync",
                "uvicorn",
                backend,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
        else:
            server_command = [
                sys.executable,
                "-m",
                "uvicorn",
                backend,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]

        print(f"Starting {backend} at {url}", flush=True)

        with open(Path(tmp) / "server.log", "w") as server_log:
            server = subprocess.Popen(
                server_command,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )

            try:
                wait_for_server(url, server)

                notebook_command = (
                    ["uv", "run", "--no-sync", "jupyter"]
                    if args.runner == "uv"
                    else [sys.executable, "-m", "jupyter"]
                )

                notebook_command += [
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(notebook),
                    "--output",
                    notebook.name,
                    "--output-dir",
                    tmp,
                    "--ExecutePreprocessor.timeout=600",
                ]

                print(f"Executing {notebook}", flush=True)
                result = subprocess.run(notebook_command, env=env)

                if result.returncode != 0:
                    print(
                        (Path(tmp) / "server.log").read_text(),
                        file=sys.stderr,
                    )

                return result.returncode

            except Exception:
                print(
                    "\n--- HTTP server log ---",
                    file=sys.stderr,
                )
                print(
                    (Path(tmp) / "server.log").read_text(),
                    file=sys.stderr,
                )
                raise

            finally:
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    sys.exit(main())
