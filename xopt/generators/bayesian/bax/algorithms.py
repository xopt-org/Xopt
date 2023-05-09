from typing import Union

import torch
from botorch.models.model import Model

from torch import Tensor


class Algorithm:
    def __init__(
        self,
        domain: Tensor,  # shape (ndim, 2)
        n_samples: int,
    ) -> None:
        self.domain = torch.tensor(domain)
        self.n_samples = n_samples
        self.ndim = domain.shape[0]


class GridScanAlgo(Algorithm):
    def __init__(
        self,
        domain: Tensor,  # shape (ndim, 2) tensor
        n_samples: int,
        n_steps_sample_grid: Union[int, list[int]],
    ) -> None:
        self.domain = domain
        self.n_samples = n_samples
        self.ndim = domain.shape[0]

        # check to see if n_steps_sample_grid is an int
        if isinstance(n_steps_sample_grid, int):
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
            mesh_points_serialized = x_mesh_columnized_tuple[0]
        else:
            mesh_points_serialized = torch.cat(x_mesh_columnized_tuple, dim=1)

        return mesh_points_serialized, x_mesh_tuple

    def eval_sample_grid_scans(self, model: Model):
        sample_xs, x_mesh_tuple = self.build_input_mesh()

        # evaluate grid scans for each posterior sample
        with torch.no_grad():
            p = model.posterior(sample_xs)
            sample_ys = p.rsample(torch.Size([self.n_samples]))

        y_mesh_samples = sample_ys.reshape(self.n_samples, *x_mesh_tuple[0].shape)

        return sample_xs, sample_ys, x_mesh_tuple, y_mesh_samples


class GridMinimize(GridScanAlgo):
    def get_exe_paths(self, model: Model):
        (
            sample_xs,
            sample_ys,
            x_mesh_tuple,
            y_mesh_samples,
        ) = self.eval_sample_grid_scans(model)

        # get exe path subsequences (in this case, just 1 (x,y) pair from each sample)
        ys_opt, min_ids = torch.min(sample_ys, dim=1)
        xs_opt = sample_xs[min_ids]

        xs_exe = xs_opt.reshape(
            -1, 1, self.ndim
        )  # xs_exe.shape = (n_samples, len_exe_path, ndim)
        ys_exe = ys_opt.reshape(-1, 1, 1)  # ys_exe.shape = (n_samples, len_exe_path, 1)

        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "sample_xs": sample_xs,
            "sample_ys": sample_ys,
            "x_mesh_tuple": x_mesh_tuple,
            "y_mesh_samples": y_mesh_samples,
        }

        return xs_exe, ys_exe, results_dict
