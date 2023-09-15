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

"""
Xopt MPI driver

Basic usage:

mpirun -n 4 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml


"""


if __name__ == "__main__":
    logger = logging.getLogger("xopt")

    # ARGS = 'xopt.in'.split()

    parser = argparse.ArgumentParser(description="Configure xopt")
    parser.add_argument("input_file", help="input_file")

    parser.add_argument("--logfile", "-l", help="Log file to write to")

    parser.add_argument("--verbose", "-v", action="count", help="Show more log output")
    parser.add_argument(
        "--asynch", "-a", action="store_true", help="Use asynchronous execution"
    )

    args = parser.parse_args()
    print(args)

    infile = args.input_file
    assert os.path.exists(infile), f"Input file does not exist: {infile}"

    level = "WARN"
    if args.verbose:
        iv = args.verbose
        if iv == 1:
            level = "WARN"
        elif iv == 2:
            level = "INFO"
        elif iv >= 3:
            level = "DEBUG"

        set_handler_with_logger(level=level)

    if args.logfile:
        set_handler_with_logger(file=args.logfile, level=level)

    # logger.info(xopt_logo)
    # logger.info('_________________________________')
    logger.info(f"Parallel execution with {mpi_size} workers")

    config = yaml.safe_load(infile)

    if args.asynch:
        logger.info("Enabling async mode")
        X = AsynchronousXopt.model_validate(config)
    else:
        X = Xopt.model_validate(config)

    print(X)
    sys.stdout.flush()
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        # with MPIPoolExecutor() as executor:

        X.evaluator.executor = executor
        X.evaluator.max_workers = mpi_size
        X.run()
