import torch


def sum_samplewise_emittance_flat_X_wrapper_for_scipy(post_paths, meas_dim, X_meas):
    def wrapped_func(X_tuning_flat):
        return (
            sum_samplewise_emittance_flat_X(
                post_paths, meas_dim, torch.tensor(X_tuning_flat), X_meas
            )
            .detach()
            .cpu()
            .numpy()
        )

    return wrapped_func


def sum_samplewise_emittance_flat_X_wrapper_for_torch(post_paths, meas_dim, X_meas):
    def wrapped_func(X_tuning_flat):
        return sum_samplewise_emittance_flat_X(
            post_paths, meas_dim, X_tuning_flat, X_meas
        )

    return wrapped_func


def sum_samplewise_emittance_flat_X(post_paths, meas_dim, X_tuning_flat, X_meas):
    X_tuning = X_tuning_flat.double().reshape(post_paths.n_samples, -1)

    return torch.sum(
        (
            post_path_emit(
                post_paths, meas_dim, X_tuning, X_meas, samplewise=True, squared=True
            )[0]
        )
    )


def post_path_emit(
    post_paths,
    meas_dim,
    X_tuning,
    X_meas,
    samplewise=False,
    squared=True,
    convert_quad_xs=True,
):
    # each row of X_tuning defines a location in the tuning parameter space,
    # along which to perform a quad scan and evaluate emit

    # X_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # if samplewise=False, X should be shape: n_tuning_configs x (ndim-1)
    # the emittance for every point specified by X will be evaluated
    # for every posterior sample path (broadcasting).

    # if samplewise=True, X must be shape: nsamples x (ndim-1)
    # the emittance of the nth sample will be computed for the nth input given by X

    # expand the X tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_quad_scan = len(
        X_meas
    )  # number of points in the scan uniformly spaced along measurement domain
    n_tuning_configs = X_tuning.shape[
        0
    ]  # the number of points in the tuning parameter space specified by X

    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas)

    if convert_quad_xs:
        k_meas = get_quad_ks_from_PV_vals(X_meas)
    else:
        k_meas = X_meas

    if samplewise:
        # add assert n_tuning_configs == post_paths.n_samples
        xs = xs.reshape(n_tuning_configs, n_steps_quad_scan, -1)
        ys = post_paths(xs)  # ys will be nsamples x n_steps_quad_scan

        (
            emits,
            emits_squared,
            abc_is_valid,
        ) = compute_emits_from_batched_beamsize_scans(k_meas, ys)[:3]

    else:
        ys = post_paths(
            xs
        )  # ys will be shape n_samples x (n_tuning_configs*n_steps_quad_scan)

        n_samples = ys.shape[0]

        ys = ys.reshape(
            n_samples * n_tuning_configs, n_steps_quad_scan
        )  # reshape into batchshape x n_steps_quad_scan

        (
            emits,
            emits_squared,
            abc_is_valid,
        ) = compute_emits_from_batched_beamsize_scans(k_meas, ys)[:3]

        emits = emits.reshape(n_samples, -1)
        emits_squared = emits_squared.reshape(n_samples, -1)
        abc_is_valid = abc_is_valid.reshape(n_samples, -1)

        # emits_flat, emits_squared_raw_flat will be tensors of
        # shape nsamples x n_tuning_configs, where n_tuning_configs
        # is the number of rows in the input tensor X.
        # The nth column of the mth row represents the emittance of the mth sample,
        # evaluated at the nth tuning config specified by the input tensor X.

    if squared:
        out = emits_squared
    else:
        out = emits

    return out, abc_is_valid


def sum_batch_post_mean_emit_flat_X_wrapper_for_scipy(
    model, meas_dim, X_meas, squared=True, batchsize=1
):
    def wrapped_func(X_tuning_flat):
        return (
            post_mean_emit(
                model,
                meas_dim,
                torch.tensor(X_tuning_flat).reshape(batchsize, -1),
                X_meas,
                squared=squared,
            )[0]
            .flatten()
            .sum()
            .detach()
            .cpu()
            .numpy()
        )

    return wrapped_func


def sum_batch_post_mean_emit_flat_X_wrapper_for_torch(
    model, meas_dim, X_meas, squared=True, batchsize=1
):
    def wrapped_func(X_tuning_flat):
        return (
            post_mean_emit(
                model,
                meas_dim,
                X_tuning_flat.reshape(batchsize, -1),
                X_meas,
                squared=squared,
            )[0]
            .flatten()
            .sum()
        )

    return wrapped_func


def post_mean_emit_flat_X_wrapper_for_scipy(model, meas_dim, X_meas, squared=True):
    def wrapped_func(X_tuning_flat):
        return (
            post_mean_emit(
                model,
                meas_dim,
                torch.tensor(X_tuning_flat).reshape(1, -1),
                X_meas,
                squared=squared,
            )[0]
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )

    return wrapped_func


def post_mean_emit_flat_X_wrapper_for_torch(model, meas_dim, X_meas, squared=True):
    def wrapped_func(X_tuning_flat):
        return post_mean_emit(
            model, meas_dim, X_tuning_flat.reshape(1, -1), X_meas, squared=squared
        )[0].flatten()

    return wrapped_func


def post_mean_emit(
    model, meas_dim, X_tuning, X_meas, squared=True, convert_quad_xs=True
):
    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas)
    ys = model.posterior(xs).mean

    ys_batch = ys.reshape(X_tuning.shape[0], -1)

    if convert_quad_xs:
        k_meas = get_quad_ks_from_PV_vals(X_meas)
    else:
        k_meas = X_meas

    (
        emits,
        emits_squared,
        abc_is_valid,
    ) = compute_emits_from_batched_beamsize_scans(
        k_meas, ys_batch
    )[:3]

    if squared:
        out = emits_squared
    else:
        out = emits

    return out, abc_is_valid


def get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas):
    # each row of X_tuning defines a location in the tuning parameter space
    # along which to perform a quad scan and evaluate emit

    # X_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # expand the X tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_meas_scan = len(X_meas)
    n_tuning_configs = X_tuning.shape[
        0
    ]  # the number of points in the tuning parameter space specified by X

    # prepare column of measurement scans coordinates
    X_meas_repeated = X_meas.repeat(n_tuning_configs).reshape(
        n_steps_meas_scan * n_tuning_configs, 1
    )

    # repeat tuning configs as necessary and concat with column from the line above
    # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
    # where d is the full dimension of the model/posterior space (tuning & meas)
    xs_tuning = torch.repeat_interleave(X_tuning, n_steps_meas_scan, dim=0)
    xs = torch.cat(
        (xs_tuning[:, :meas_dim], X_meas_repeated, xs_tuning[:, meas_dim:]), dim=1
    )

    return xs


def compute_emits_from_batched_beamsize_scans(ks_meas, ys_batch):
    # xs_meas is assumed to be a 1d tensor of shape (n_steps_quad_scan,)
    # representing the measurement parameter inputs of the emittance scan

    # ys_batch is assumed to be shape n_scans x n_steps_quad_scan,
    # where each row represents the beamsize outputs of an emittance scan
    # with input given by xs_meas

    # note that every measurement scan is assumed to have been evaluated
    # at the single set of measurement param inputs described by xs_meas

    # geometric configuration for LCLS OTR2 emittance/quad measurement scan
    q_len = 0.108  # measurement quad thickness
    distance = 2.26  # drift length from measurement quad to observation screen

    device = ks_meas.device

    ks_meas = ks_meas.reshape(-1, 1)
    xs_meas = ks_meas * distance * q_len

    # least squares method to calculate parabola coefficients
    A_block = torch.cat(
        (
            xs_meas**2,
            xs_meas,
            torch.tensor([1], device=device)
            .repeat(len(xs_meas))
            .reshape(xs_meas.shape),
        ),
        dim=1,
    )
    B = ys_batch.double()

    #     print('B.shape =', B.shape)
    #     print('A_block.shape =', A_block.shape)
    #     print('A.shape =', A.shape)
    abc = A_block.pinverse().repeat(*ys_batch.shape[:-1], 1, 1).double() @ B.reshape(
        *B.shape, 1
    )
    abc = abc.reshape(*abc.shape[:-1])
    abc_is_valid = torch.logical_and(
        abc[:, 0] > 0, (abc[:, 2] > abc[:, 1] ** 2 / (4.0 * abc[:, 0]))
    )

    # analytically calculate the Sigma (beam) matrices from parabola coefficients
    # (non-physical results are possible)
    M = torch.tensor(
        [
            [1, 0, 0],
            [-1 / distance, 1 / (2 * distance), 0],
            [1 / (distance**2), -1 / (distance**2), 1 / (distance**2)],
        ],
        device=device,
    )

    sigs = torch.matmul(
        M.repeat(*abc.shape[:-1], 1, 1).double(),
        abc.reshape(*abc.shape[:-1], 3, 1).double(),
    )  # column vectors of sig11, sig12, sig22

    Sigmas = (
        sigs.reshape(-1, 3)
        .repeat_interleave(torch.tensor([1, 2, 1], device=device), dim=1)
        .reshape(*sigs.shape[:-2], 2, 2)
    )  # 2x2 Sigma/covar beam matrix

    # compute emittances from Sigma (beam) matrices
    emits_squared_raw = torch.linalg.det(Sigmas)

    emits = torch.sqrt(
        emits_squared_raw
    )  # these are the emittances for every tuning parameter combination.

    emits_squared_raw = emits_squared_raw.reshape(ys_batch.shape[0], -1)
    emits = emits.reshape(ys_batch.shape[0], -1)

    return emits, emits_squared_raw, abc_is_valid, abc


def get_valid_emittance_samples(
    model, domain, meas_dim, X_tuning, n_samples=10000, n_steps_quad_scan=10
):
    x_meas = torch.linspace(*domain[meas_dim], n_steps_quad_scan)
    xs_1d_scan = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, x_meas)

    p = model.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)

    ks = get_quad_ks_from_PV_vals(x_meas)
    (
        emits_at_target,
        emits_sq_at_target,
        is_valid,
    ) = compute_emits_from_batched_beamsize_scans(ks, bss)[:3]
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq_at_target.shape[0]))[is_valid]
    emits_sq_at_target_valid = torch.index_select(
        emits_sq_at_target, dim=0, index=cut_ids
    )
    emits_at_target_valid = emits_sq_at_target_valid.sqrt()
    return emits_at_target_valid, emits_sq_at_target, is_valid, sample_validity_rate


def get_quad_ks_from_PV_vals(xs_quad, E=0.135, q_len=0.108):
    E = 0.135  # beam energy in GeV
    gamma = E / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)

    quad_field_integrals = xs_quad / 10.0  # divide by 10 to convert from kG to Tesla
    gs_quad = quad_field_integrals / q_len  # divide by thickness to get gradients
    ks_quad = 0.299 * gs_quad / (beta * E)  # quadrupole geometric focusing strength

    return ks_quad
