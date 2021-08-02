import torch
import numpy as np


def standardize(Y):
    # check if there are nans -> if there are we cannot use gradients
    if torch.any(torch.isnan(Y)):
        stddim = -1 if Y.dim() < 2 else -2
        stddim = -2
        Y_np = Y.detach().numpy()

        # NOTE differences between std calc for torch and numpy to get unbiased estimator
        # see aboutdatablog.com/post/why-computing-standard-deviation-in-
        # pandas-and-numpy-yields-different-results
        std = np.nanstd(Y_np, axis=stddim, keepdims=True, ddof=1)
        std = np.where(std >= 1e-9, std, np.full_like(std, 1.0))

        Y_std_np = (Y_np - np.nanmean(Y_np, axis=stddim, keepdims=True)) / std
        return torch.tensor(Y_std_np, dtype=Y.dtype)

    else:
        stddim = -1 if Y.dim() < 2 else -2
        Y_std = Y.std(dim=stddim, keepdim=True)
        Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
        return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std


if __name__ == '__main__':
    t = torch.tensor(((1., 2., 3.), (4., 5., 6.), (7., 8., 9.)))
    print(t)
    t[0, 0] = np.nan
    print(standardize(t))
