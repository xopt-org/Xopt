import argparse
import logging
import os
import sys

import yaml
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from xopt import AsynchronousXopt

# from mpi4py.futures import MPIPoolExecutor
from xopt.base import Xopt
from xopt.log import set_handler_with_logger

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

logger = logging.getLogger("xopt")


def run_mpi(config, verbosity=None, asynchronous=True, logfile=None):
    """
    Xopt MPI driver

    Basic usage:

    mpirun -n 4 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml


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
        set_handler_with_logger(file=args.logfile, level=level)

    # logger.info(xopt_logo)
    # logger.info('_________________________________')
    logger.info(f"Parallel execution with {mpi_size} workers")

    if asynchronous:
        logger.info("Enabling async mode")
        X = AsynchronousXopt(**config)
    else:
        X = Xopt(**config)

    print(X)
    sys.stdout.flush()
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        # with MPIPoolExecutor() as executor:

        X.evaluator.executor = executor
        X.evaluator.max_workers = mpi_size
        X.run()


if __name__ == "__main__":
    # ARGS = 'xopt.in'.split()

    parser = argparse.ArgumentParser(description="Configure xopt")
    parser.add_argument("input_file", help="input_file")

    parser.add_argument("--logfile", "-l", help="Log file to write to")

    parser.add_argument("--verbose", "-v", action="count", help="Show more log output")
    parser.add_argument(
        "--asynchronous",
        "-a",
        action="store_true",
        help="Use asynchronous execution",
        default=True,
    )

    args = parser.parse_args()
    print(args)

    input_file = args.input_file
    logfile = args.logfile
    verbosity = args.verbose
    asynchronous = args.asynchronous

    assert os.path.exists(input_file), f"Input file does not exist: {input_file}"

    config = yaml.safe_load(open(input_file))

    run_mpi(config, verbosity=verbosity, logfile=logfile, asynchronous=asynchronous)
