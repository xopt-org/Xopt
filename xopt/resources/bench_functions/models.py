import logging

import gpytorch
import numpy as np
import torch

from xopt.generators.bayesian.utils import get_training_data_batched
from xopt.resources.benchmarking import BenchDispatcher, generate_data, generate_vocs
from xopt.generators.bayesian.models.standard import (
    BatchedModelConstructor,
    StandardModelConstructor,
)

logging.basicConfig(level=logging.DEBUG)


def bench_build_standard_kwargs():
    test_vocs = generate_vocs(n_vars=12, n_obj=5, n_constr=2)
    test_data = generate_data(vocs=test_vocs, n=100)
    return test_vocs, test_data


def preamble_build_model():
    # import numpy as np
    # np.show_runtime()
    torch.set_num_threads(1)
    # botorch already uses same method internally, but force globally for other libs
    import threadpoolctl
    from pprint import pprint

    threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
    pprint(threadpoolctl.threadpool_info())

    torch.cuda.synchronize()


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_standard(vocs, data, device="cpu"):
    device = torch.device(device)
    gp_constructor = StandardModelConstructor()
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_standard_adam(vocs, data, device="cpu"):
    device = torch.device(device)
    gp_constructor = StandardModelConstructor(method="adam")
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_batched(vocs, data, device="cpu"):
    device = torch.device(device)
    gp_constructor = BatchedModelConstructor()
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_batched_botorch_patch(vocs, data, device="cpu"):
    def as_ndarray(values, dtype=None, inplace=True):
        if inplace:
            return values.numpy(force=True).astype(dtype, copy=False)
        else:
            return values.clone().numpy(force=True).astype(dtype, copy=False)

    import botorch.optim.closures as closures

    closures.core.as_ndarray = as_ndarray
    device = torch.device(device)
    gp_constructor = BatchedModelConstructor()
    model = gp_constructor.build_model_from_vocs(vocs, data, device=device)
    return model


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_batched_gpytorch(vocs, data, device="cpu"):
    class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([vocs.n_outputs])
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([vocs.n_outputs])),
                batch_shape=torch.Size([vocs.n_outputs]),
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )

    train_X, train_Y, train_Yvar = get_training_data_batched(
        input_names=vocs.variable_names,
        outcome_names=vocs.output_names,
        data=data,
        batch_mode=False,
    )
    device = torch.device(device)
    # disabling global noise is critical to match list implementation
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=vocs.n_outputs, has_global_noise=False
    )
    model = BatchIndependentMultitaskGPModel(train_X, train_Y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    model.to(device=device)
    mll.to(device=device)
    training_iterations = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #print([(p, p.shape) for p in model.parameters()])
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        if i % 10 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    # [10.246825   8.605122   6.923973   5.161834   4.2145376  4.3885226  4.4575253]
    print(
        "Lengthscale: ",
        model.covar_module.base_kernel.raw_lengthscale.numpy(force=True).flatten(),
    )


@BenchDispatcher.register_decorator(preamble=preamble_build_model)
@BenchDispatcher.register_defaults(
    ["vocs", "data"], lambda: bench_build_standard_kwargs()
)
def bench_build_standard_gpytorch(vocs, data, device="cpu"):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    train_X, train_Y, train_Yvar = get_training_data_batched(
        input_names=vocs.variable_names,
        outcome_names=vocs.output_names,
        data=data,
        batch_mode=False,
    )

    device = torch.device(device)
    torch.use_deterministic_algorithms(True)
    models = []
    for i in range(vocs.n_outputs):
        models.append(
            ExactGPModel(
                train_X,
                train_Y[:, i],
                gpytorch.likelihoods.GaussianLikelihood(),
            )
        )
    model = gpytorch.models.IndependentModelList(*models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[m.likelihood for m in models])
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    model.to(device=device)
    mll.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #print([(p, p.shape) for p in model.parameters()])
    training_iterations = 200
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        if i % 10 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    ls = np.hstack(
        [
            m.covar_module.base_kernel.raw_lengthscale.numpy(force=True).flatten()
            for m in models
        ]
    )
    # [10.246825   8.605122   6.9239607  5.161834   4.214438   4.3885236   4.4575253]
    print("Lengthscale: ", ls)
