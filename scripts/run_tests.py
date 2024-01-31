#!/usr/bin/env python
import sys

import pytest

if __name__ == "__main__":
    # Show output results from every test function
    # Show the message output for skipped and expected failures
    args = ["-v", "-vrxs", "--ignore=benchmarks"]

    # Add extra arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])

    # Show coverage
    if "--show-cov" in args:
        args.extend(["--cov=xopt", "--cov-report", "term-missing"])
        args.remove("--show-cov")

    print("pytest arguments: {}".format(args))
    print(f"Running tests on Python {sys.version}")
    sys.exit(pytest.main(args))
