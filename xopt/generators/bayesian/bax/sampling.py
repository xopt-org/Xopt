import copy
import math

import gpytorch
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths


def draw_poly_kernel_prior_paths(
    poly_kernel, n_samples
):  # poly_kernel is a scaled polynomial kernel
    c = poly_kernel.offset
    degree = poly_kernel.power
    ws = torch.randn(size=[n_samples, 1, degree + 1])

    def paths(xs):
        if (
            len(xs.shape) == 2 and xs.shape[1] == 1
        ):  # xs must be n_samples x npoints x 1 dim
            xs = xs.repeat(n_samples, 1, 1)  # duplicate over batch (sample) dim

        coeffs = [math.comb(degree, i) for i in range(degree + 1)]
        X = torch.concat(
            [
                (coeff * c.pow(i)).sqrt() * xs.pow(degree - i)
                for i, coeff in enumerate(coeffs)
            ],
            dim=2,
        )
        W = ws.repeat(1, xs.shape[1], 1)  # ws is n_samples x 1 x 3 dim

        phis = W * X
        return torch.sum(phis, dim=-1)  # result tensor is shape n_samples x npoints

    return paths


def draw_quad_kernel_prior_paths(
    quad_kernel, n_samples
):  # quad_kernel is a scaled polynomial(power=2) kernel
    c = quad_kernel.offset
    ws = torch.randn(size=[n_samples, 1, 3])

    def paths(xs):
        if (
            len(xs.shape) == 2 and xs.shape[1] == 1
        ):  # xs must be n_samples x npoints x 1 dim
            xs = xs.repeat(n_samples, 1, 1)  # duplicate over batch (sample) dim

        X = torch.concat([xs * xs, (2 * c).sqrt() * xs, c.expand(*xs.shape)], dim=2)
        W = ws.repeat(1, xs.shape[1], 1)  # ws is n_samples x 1 x 3 dim

        phis = W * X
        return torch.sum(phis, dim=-1)  # result tensor is shape n_samples x npoints

    return paths


def draw_product_kernel_prior_paths(model, n_samples):
    ndim = model.train_inputs[0].shape[1]

    matern_covar_module = copy.deepcopy(
        model.covar_module.base_kernel.kernels[0]
    )  # expects ProductKernel (Matern x Polynomial(dim=2))
    matern_dims = copy.copy(model.covar_module.base_kernel.kernels[0].active_dims)
    # add assert matern
    #     print('matern_covar_module.active_dims =', matern_covar_module.active_dims)
    matern_covar_module.active_dims = None
    matern_covar_module = gpytorch.kernels.ScaleKernel(matern_covar_module)
    matern_covar_module.outputscale = copy.copy(model.covar_module.outputscale.detach())

    mean_module = gpytorch.means.ZeroMean()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cpu()
    likelihood.noise = copy.copy(model.likelihood.noise.detach())

    outcome_transform = None
    input_transform = None

    # build zero-mean (ndim-1)-dimensional GP called matern_model
    # with kernel matched to the Matern component of the passed model

    matern_model = SingleTaskGP(
        train_X=torch.tensor([[0.0] * (ndim - 1)]),  # add index specification
        train_Y=torch.tensor([[0.0]]),
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=matern_covar_module,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
    )

    matern_prior_paths = draw_kernel_feature_paths(
        model=matern_model, sample_shape=torch.Size([n_samples])
    )

    quad_kernel = copy.deepcopy(model.covar_module.base_kernel.kernels[1])
    quad_dim = copy.copy(quad_kernel.active_dims)

    # add assert polynomial kernel pow=2 with 1 active dim

    quad_prior_paths = draw_poly_kernel_prior_paths(quad_kernel, n_samples)

    def product_kernel_prior_paths(xs):
        xs_matern = torch.index_select(xs, dim=-1, index=matern_dims).float()
        xs_quad = torch.index_select(xs, dim=-1, index=quad_dim).float()
        return (
            matern_prior_paths(xs_matern).reshape(n_samples, -1)
            * quad_prior_paths(xs_quad)
        ).double()

    return product_kernel_prior_paths


def draw_product_kernel_post_paths(model, n_samples, cpu=True):
    product_kernel_prior_paths = draw_product_kernel_prior_paths(
        model, n_samples=n_samples
    )

    train_x = model.train_inputs[0]

    train_y = model.train_targets.reshape(-1, 1)

    train_y = train_y - model.mean_module(train_x).reshape(train_y.shape)

    Knn = model.covar_module(train_x, train_x)

    sigma = torch.sqrt(model.likelihood.noise[0])

    K = Knn + sigma**2 * torch.eye(Knn.shape[0])

    prior_residual = train_y.repeat(n_samples, 1, 1).reshape(
        n_samples, -1
    ) - product_kernel_prior_paths(train_x)
    prior_residual -= sigma * torch.randn_like(prior_residual)

    Lnn = torch.cholesky(K.to_dense())
    batched_lnn = torch.stack([Lnn] * n_samples)
    batched_lnnt = torch.stack([Lnn.T] * n_samples)

    vbar = torch.linalg.solve(batched_lnn, prior_residual)
    v = torch.linalg.solve(batched_lnnt, vbar)
    v = v.reshape(-1, 1)

    v = v.reshape(n_samples, -1, 1)
    v_t = v.transpose(1, 2)

    def post_paths(xs):
        if model.input_transform is not None:
            xs = model.input_transform(xs)

        K_update = model.covar_module(train_x, xs.double())

        update = torch.matmul(v_t, K_update)
        update = update.reshape(n_samples, -1)

        prior = product_kernel_prior_paths(xs)

        post = prior + update + model.mean_module(xs)
        if model.outcome_transform is not None:
            post = model.outcome_transform.untransform(post)[0]

        return post

    post_paths.n_samples = n_samples

    return post_paths


def compare_sampling_methods(
    model, domain, scan_dim, n_samples_per_batch=100, n_batches=100, verbose=False
):
    from matplotlib import pyplot as plt

    ndim = model.train_inputs[0].shape[-1]
    xs_1d_scan = torch.zeros(100, ndim)

    xlin = torch.linspace(*domain[scan_dim], 100)

    xs_1d_scan[:, scan_dim] = xlin

    all_pathwise_samples = torch.Tensor([])
    all_standard_samples = torch.Tensor([])

    for i in range(n_batches):
        if verbose:
            print("batch", i)

        # pathwise sampling
        post_paths = draw_product_kernel_post_paths(
            model, n_samples=n_samples_per_batch
        )

        pathwise_samples = post_paths(xs_1d_scan).detach()

        # standard sampling
        p = model.posterior(xs_1d_scan)
        s = p.rsample(torch.Size([n_samples_per_batch]))
        standard_samples = s.reshape(n_samples_per_batch, -1).detach()

        # cat results
        all_pathwise_samples = torch.cat(
            (all_pathwise_samples, pathwise_samples), dim=0
        )
        all_standard_samples = torch.cat(
            (all_standard_samples, standard_samples), dim=0
        )

    fig, ax = plt.subplots(1)

    ax.fill_between(
        xlin.squeeze().cpu(),
        torch.quantile(all_standard_samples, 0.025, dim=0).cpu(),
        torch.quantile(all_standard_samples, 0.975, dim=0).cpu(),
        color="C0",
        alpha=0.3,
        label="Built-in Posterior Sampling 95% CI",
    )
    ax.plot(
        xlin.squeeze().cpu(),
        torch.mean(all_standard_samples, dim=0).cpu(),
        color="C0",
        linewidth=5,
        label="Built-in Posterior Mean",
    )

    ax.plot(
        xlin.squeeze().cpu(),
        torch.quantile(all_pathwise_samples, 0.975, dim=0).cpu(),
        c="k",
        linestyle="--",
        label="Pathwise Posterior Sampling 95% CI",
    )
    ax.plot(
        xlin.squeeze().cpu(),
        torch.quantile(all_pathwise_samples, 0.025, dim=0).cpu(),
        c="k",
        linestyle="--",
    )

    ax.plot(
        xlin.squeeze().cpu(),
        torch.mean(all_pathwise_samples, dim=0).cpu(),
        color="k",
        label="Pathwise Sampling Posterior Mean",
    )

    # ax.scatter(x_obs, y_obs, c='k', label='Observations')
    ax.legend(loc="upper center")

    ax.set_ylabel("Output")
    ax.set_title("Product Kernel Posterior Sampling Comparison")

    scan_param = "$x_{{{0}}}$".format(str(scan_dim))

    xlabel = scan_param

    ax.set_xlabel(xlabel)
    textstr = r"$ndim=%d$" % (int(xs_1d_scan.shape[1]),)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()
    plt.close()
