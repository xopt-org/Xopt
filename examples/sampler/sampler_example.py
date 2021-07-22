# main function
from xopt.sampler import random_sampler

# test function
from xopt.evaluators import test_TNK

if __name__ == '__main__':
    # Get VOCS
    VOCS = test_TNK.VOCS
    print(VOCS)

    # Get evaluate function
    EVALUATE = test_TNK.evaluate_TNK
    print(EVALUATE)

    #VOCS['variables']['x1'] = [0, 4]  # Extent to occasionally throw an exception

    from concurrent.futures import ProcessPoolExecutor
    from tempfile import TemporaryDirectory

    SCRATCH = TemporaryDirectory()

    executor = ProcessPoolExecutor()

    # Run
    results = random_sampler(executor, vocs=VOCS, evaluate_f=EVALUATE, verbose=True, output_path=SCRATCH.name)
