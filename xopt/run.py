#!/usr/bin/env python
# coding: utf-8


"""
xopt driver using Processes

Basic usage:

python -m xopt.run xopt.yaml

positional arguments:
  input_file        Xopt YAML input file

optional arguments:
  -h, --help        show this help message and exit
  -n MAX_WORKERS    Maximum workers
  -logfile LOGFILE  log file
  -v, --verbose     Verbosity level -v (INFO), -vv (DEBUG)


"""

from xopt import Xopt, xopt_logo

import logging


from concurrent.futures import ProcessPoolExecutor

import argparse
import os
import sys

from psutil import cpu_count

from pprint import pprint


#ARGS = sys.argv[1:]
#ARGS = 'xopt.in'.split()

parser = argparse.ArgumentParser(description='Configure xopt')

# Main input file
parser.add_argument('input_file', help='Xopt YAML input file')

# Max workers for processes. 
parser.add_argument('-n', dest='max_workers', default=None, help='Maximum workers')

# Logging arguments
parser.add_argument('-logfile', dest='logfile', default=None, help='log file')

# Logging Verbosity levels 
parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity level -v (INFO), -vv (DEBUG) ') 



# Parse args
args = parser.parse_args()

# handle -v, -vv logging levels
levels = [logging.WARNING, logging.INFO, logging.DEBUG]
level = levels[min(len(levels)-1, args.verbose)]  # capped to number of levels

# Setup logging
logging.basicConfig(filename=args.logfile,                   
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                    level=level
                   )
logger = logging.getLogger(__name__)


logger.debug("a debug message")
logger.info("a info message")
logger.warning("a warning message")

infile = args.input_file
assert os.path.exists(infile), f'Input file does not exist: {infile}'

if __name__ == "__main__":
    print(xopt_logo)
    print('_________________________________')
    print('Parallel execution with processes')   
    if args.max_workers:
        print('max_workers', args.max_workers)
    else:
        print('Automatic max workers')
    
    logger.info(f'CPU count: {cpu_count()}')
    logger.debug('this is a debug message')        
        
    X = Xopt(infile)
    print(X)
    sys.stdout.flush() 
    with ProcessPoolExecutor() as executor:
        X.run(executor=executor)
