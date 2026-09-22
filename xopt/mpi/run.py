from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import argparse
import logging
import os
import sys
import yaml

from xopt import AsynchronousXopt
from xopt.base import Xopt
from xopt.entrypoint import merge_dicts, override_to_dict, setup_import_paths
from xopt.log import set_handler_with_logger

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

logger = logging.getLogger("xopt")


def run_mpi(config, verbosity=None, asynchronous=True, logfile=None, python_path=None):
    """
    Xopt MPI driver

    Basic usage:

    mpirun -n 4 xopt-mpirun xopt.yaml

    Parameters
    ----------
    config : dict
        Xopt configuration.
    verbosity : int, optional
        Verbosity count controlling the log level.
    asynchronous : bool, default=True
        Run with AsynchronousXopt instead of Xopt.
    logfile : str, optional
        File log messages are written to.
    python_path : list of str, optional
        Directories added to the module search path on every rank. Worker ranks import
        the evaluation function themselves, so this must run before the Xopt object is
        built.
    """

    level = "WARN"
    if verbosity:
        iv = verbosity
        if iv == 1:
            level = "WARN"
        elif iv == 2:
            level = "INFO"
        elif iv >= 3:
            level = "DEBUG"

        set_handler_with_logger(level=level)

    if logfile:
        set_handler_with_logger(file=logfile, level=level)

    setup_import_paths(python_path or [])

    # logger.info(xopt_logo)
    # logger.info('_________________________________')
    logger.info(f"Parallel execution with {mpi_size} workers")

    if asynchronous:
        logger.info("Enabling async mode")
        X = AsynchronousXopt(**config)
    else:
        X = Xopt(**config)

    if mpi_rank == 0:
        print(X)
        sys.stdout.flush()

    # Launch MPI executor (it is None for the workers)
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            X.evaluator.executor = executor
            X.evaluator.max_workers = mpi_size
            X.run()


def main():
    parser = argparse.ArgumentParser(description="Configure xopt")
    parser.add_argument("input_file", help="input_file")
    parser.add_argument("--logfile", "-l", help="Log file to write to")
    parser.add_argument("--verbose", "-v", action="count", help="Show more log output")
    parser.add_argument(
        "--asynchronous",
        "-a",
        action=argparse.BooleanOptionalAction,
        help="Use asynchronous execution",
        default=True,
    )
    parser.add_argument(
        "--python_path",
        help="Additional path to add to Python import path for evaluation function module search",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--override",
        help="Override config values using dot notation (e.g., generator.mutation_operator.eta_m=20)",
        action="append",
        default=[],
    )

    args = parser.parse_args()
    if mpi_rank == 0:
        print(args)

    input_file = args.input_file
    logfile = args.logfile
    verbosity = args.verbose
    asynchronous = args.asynchronous

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        exit()

    config = yaml.safe_load(open(input_file))

    if args.override:
        logger.info("Applying config file overrides:")
    for override in args.override:
        logger.info(f"  {override}")
        if "=" not in override:
            raise ValueError(
                f'Invalid override format: "{override}". Expected key=value'
            )
        config = merge_dicts(config, override_to_dict(override))

    run_mpi(
        config,
        verbosity=verbosity,
        logfile=logfile,
        asynchronous=asynchronous,
        python_path=[os.getcwd()] + args.python_path,
    )


if __name__ == "__main__":
    main()
