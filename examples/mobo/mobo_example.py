# main function
import numpy as np
import torch
import matplotlib.pyplot as plt

from xopt.mobo import mobo
from botorch.test_functions.multi_objective import ZDT1

# test function
from xopt.evaluators import test_TNK
from xopt.evaluators import test_ZDT

if __name__ == '__main__':
    # Get VOCS
    VOCS = test_ZDT.VOCS

    # add reference point
    ref = torch.tensor((11., 11.))

    print(VOCS)
    # Get evaluate function
    EVALUATE = test_ZDT.evaluate

    # VOCS['variables']['x1'] = [0, 4]  # Extent to occasionally throw an exception

    # Run
    train_x, train_y, train_c, model = mobo(VOCS, EVALUATE, ref,
                                            n_steps=20, verbose=True, return_model=True)

    # plot model
    n = 30
    x = np.linspace(0, 1, n)
    xx = np.meshgrid(x, x)
    pts = torch.tensor(np.vstack((ele.ravel() for ele in xx)).T).float()

    prob = ZDT1(2)

    with torch.no_grad():
        pos = model(pts)
        mean = pos.mean
        var = pos.variance

        true = torch.transpose(prob.evaluate_true(pts), 0, 1)

    print(true.shape)
    fig2, ax2 = plt.subplots()
    c = ax2.pcolor(*xx, mean[1].reshape(n, n))
    fig2.colorbar(c)

    fig2, ax2 = plt.subplots()
    c = ax2.pcolor(*xx, torch.sqrt(var[1].reshape(n, n)))
    fig2.colorbar(c)

    fig, ax = plt.subplots()
    ax.plot(train_y[:, 0], train_y[:, 1], '.')

    plt.show()
