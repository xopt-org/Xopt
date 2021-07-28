import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.utils.objective import apply_constraints_nonnegative_soft, apply_constraints
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qUpperConfidenceBound, GenericMCObjective, UpperConfidenceBound, \
    ScalarizedObjective, MCAcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt


from acquisition.exploration import qBayesianExploration, BayesianExploration

train_X = torch.rand(30, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
train_Y = standardize(Y)

C = -((train_X[:, 0]).reshape(-1, 1) * 2.0 - 1)
print(C)

train_outputs = torch.hstack([train_Y, C])

mc_obj = GenericMCObjective(lambda Z, X: Z[..., 0])
weights = torch.tensor((1.0, 0.0))
ana_obj = ScalarizedObjective(weights)

gp = SingleTaskGP(train_X, train_outputs)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# MC acq functions
qBE = qBayesianExploration(gp, constraints=[lambda Z: Z[..., -1]], objective=mc_obj)
mc_acq = qBE

# analytical acq functions
#UCB = UpperConfidenceBound(gp, beta=100000., objective=ana_obj)
#constr = constrained.ConstrainedAcquisitionFunction(gp, {1: [-100000, 0.0]})
#acq = combine_acquisition.MultiplyAcquisitionFunction(gp, [UCB, constr])

BE = BayesianExploration(gp, 0, constraints={1: [-100000, 0.0]}, sigma=0.01*torch.eye(2))
an_acq = BE

test_val = torch.rand(20, 5, 2)
qBE.forward(test_val)

n = 100
x = np.linspace(0, 1, n)
xx = np.meshgrid(x, x)
pts = np.vstack((ele.ravel() for ele in xx)).T
pts = torch.tensor(pts).float().unsqueeze(dim=1)
print(pts.shape)

with torch.no_grad():
    mc_acq_val = mc_acq.forward(pts)
    an_acq_val = an_acq.forward(pts)

fig, ax = plt.subplots(2, 1)
ax[0].pcolor(xx[0], xx[1], mc_acq_val.reshape(n, n))
ax[0].plot(train_X.T[0], train_X.T[1], 'oC1')
ax[1].pcolor(xx[0], xx[1], an_acq_val.reshape(n, n))
ax[1].plot(train_X.T[0], train_X.T[1], 'oC1')
plt.show()
bounds = torch.stack([torch.zeros(2), torch.ones(2)])
# candidate, acq_value = optimize_acqf(
#    acq, bounds=bounds, q=5, num_restarts=5, raw_samples=20,
# )
# print(candidate)
