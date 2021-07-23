# main function
import numpy as np
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

from xopt.mobo import mobo

# test function
from xopt.evaluators import test_TNK

if __name__ == '__main__':
    # Get VOCS
    VOCS = test_TNK.VOCS

    # add reference point
    ref = torch.tensor((1.4, 1.4))

    print(VOCS)
    # Get evaluate function
    EVALUATE = test_TNK.evaluate_TNK

    # VOCS['variables']['x1'] = [0, 4]  # Extent to occasionally throw an exception

    # Run
    init_x = torch.tensor([[0.9, 0.9], [0.6, 0.6]])

    executor = PoolExecutor(max_workers=2)
    train_x, train_y, train_c = mobo(VOCS, EVALUATE, ref,
                                     batch_size=1,
                                     #executor=executor,
                                     n_initial_samples=10,
                                     mc_samples=128,
                                     initial_x=None,
                                     use_gpu=False,
                                     n_steps=50,
                                     verbose=True,
                                     plot_acq=True)

    fig, ax = plt.subplots()
    ax.plot(train_y[:, 0], train_y[:, 1], '.')

    plt.show()
