from typing import Union

import torch
from botorch.models.model import Model

from torch import Tensor


class Algorithm:
    def __init__(
        self,
        n_samples: int,
        domain: Tensor,  # shape (ndim, 2)
    ) -> None:
        self.n_samples = n_samples
        self.domain = torch.tensor(domain)
        self.ndim = domain.shape[0]

    def unif_random_sample_domain(self, n_samples, domain):
        ndim = len(domain)
        x_samples = torch.rand(n_samples, ndim) * torch.tensor(
            [bounds[1] - bounds[0] for bounds in domain]
        ) + torch.tensor(
            [bounds[0] for bounds in domain]
        )  # uniform sample, rescaled, and shifted to cover the domain

        return x_samples


class GridScanAlgo(Algorithm):
    def __init__(
        self,
        domain: Tensor,  # shape (ndim, 2) tensor
        n_samples: int,
        n_steps_sample_grid: Union[int, list[int]],
    ) -> None:
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples = n_samples

        if isinstance(
            n_steps_sample_grid, int
        ):  # check to see if n_steps_sample_grid is an int
            # if so, we make it a list with that integer repeated for every dimension
            n_steps_sample_grid = [n_steps_sample_grid] * int(self.ndim)

        if len(n_steps_sample_grid) != self.ndim:
            raise ValueError(
                "If n_steps_sample_grid is a list, it must have length = ndim"
            )

        # a list of ints specifying the number of steps per dimension in the sample grid
        self.n_steps_sample_grid = n_steps_sample_grid

    def build_input_mesh(self):
        linspace_list = [
            torch.linspace(bounds[0], bounds[1], n_steps).double()
            for n_steps, bounds in zip(self.n_steps_sample_grid, self.domain)
        ]

        x_mesh_tuple = torch.meshgrid(*linspace_list, indexing="ij")

        x_mesh_columnized_tuple = tuple(
            x_mesh.reshape(-1, 1) for x_mesh in x_mesh_tuple
        )

        if self.ndim == 1:
            xs_n_by_d = x_mesh_columnized_tuple[0]
        else:
            xs_n_by_d = torch.cat(x_mesh_columnized_tuple, dim=1)

        return xs_n_by_d, x_mesh_tuple

    def eval_sample_grid_scans(self, model: Model):
        sample_xs, x_mesh_tuple = self.build_input_mesh()

        # evaluate grid scans for each posterior sample
        with torch.no_grad():
            p = model.posterior(sample_xs)
            sample_ys = p.rsample(torch.Size([self.n_samples]))

        y_mesh_samples = sample_ys.reshape(self.n_samples, *x_mesh_tuple[0].shape)

        self.sample_xs = sample_xs
        self.sample_ys = sample_ys
        self.x_mesh_tuple = x_mesh_tuple
        self.y_mesh_samples = y_mesh_samples

        return sample_xs, sample_ys, x_mesh_tuple, y_mesh_samples


class GridOpt(GridScanAlgo):
    def get_exe_paths(self, model: Model):
        sample_xs, sample_ys = self.eval_sample_grid_scans(model)[:2]

        # get exe path subsequences (in this case, just 1 (x,y) pair from each sample)
        ys_opt, min_ids = torch.min(sample_ys, dim=1)
        xs_opt = sample_xs[min_ids]

        self.xs_exe = xs_opt.reshape(
            -1, 1, self.ndim
        )  # xs_exe.shape = (n_samples, len_exe_path, ndim)
        self.ys_exe = ys_opt.reshape(
            -1, 1, 1
        )  # ys_exe.shape = (n_samples, len_exe_path, 1)

        return self.xs_exe, self.ys_exe
