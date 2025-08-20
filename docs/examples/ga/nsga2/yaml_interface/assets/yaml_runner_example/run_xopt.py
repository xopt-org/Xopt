"""
run_xopt.py - Simple Xopt runner script
This script reads the Xopt object from a YAML file and runs it.
"""

import xopt
import argparse


if __name__ == "__main__":
    # Handle the CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="The Xopt YAML config file")
    args = parser.parse_args()

    # Create xopt
    with open(args.config) as f:
        my_xopt = xopt.Xopt.from_yaml(f.read())

    # Run it
    my_xopt.run()
