import torch
from botorch.acquisition.multi_objective.analytic import (
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.models.model import Model, ModelList
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from xopt.generators.bayesian.bax.algorithms import Algorithm


class ModelListExpectedInformationGain(MultiObjectiveAnalyticAcquisitionFunction):
    r"""Single-outcome expected information gain for independent
        multi-output (ModelListGP) models.

    Example:
        >>> model1 = SingleTaskGP(train_X, train_Y1)
        >>> model2 = SingleTaskGP(train_X, train_Y2)
        >>> model = ModelList(model1, model2)
        >>> EIG = ExpectedInformationGain(model, algo)
        >>> eig = EIG(test_X)

        Parameters
        ----------
            model: A fitted independent multi-output (ModelList) model.
    """

    def __init__(self, model: Model, algorithm: Algorithm, bounds: Tensor) -> None:
        super().__init__(model=model)
        self.algorithm = algorithm

        # get sample-wise algorithm execution (BAX) results
        (
            self.xs_exe,
            self.ys_exe,
            self.algorithm_results,
        ) = self.algorithm.get_execution_paths(self.model, bounds)

        # Need to call the model on some data before we can condition_on_observations
        self.model(*[self.xs_exe[:1, 0:1, 0:] for m in model.models])

        # construct a batch of size n_samples fantasy models,
        # where each fantasy model is produced by taking the model
        # at the current iteration and conditioning it
        # on one of the sampled execution path subsequences:
        xs_exe_t = [
            model.models[i].input_transform(self.xs_exe)
            for i in range(len(model.models))
        ]
        ys_exe_t = [
            model.models[i].outcome_transform(
                torch.index_select(self.ys_exe, dim=-1, index=torch.tensor([i]))
            )[0]
            for i in range(len(model.models))
        ]
        fantasy_models = [
            m.condition_on_observations(x, y)
            for m, x, y in zip(model.models, xs_exe_t, ys_exe_t)
        ]
        self.fantasy_models = ModelList(*fantasy_models)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Information Gain on the candidate set X.

        Parameters
        ----------
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Information Gain is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns
        -------
            A `(b1 x ... bk)`-dim tensor of Expected Information Gain values at the
            given design points `X`.
        """

        # Use the current & fantasy models to compute a
        # Monte-Carlo estimate of the Expected Information Gain:
        # see https://arxiv.org/pdf/2104.09460.pdf:
        # eq (4) and the last sentence of page 7)

        # calculcate the variance of the posterior for each input x
        post = self.model.posterior(X)
        var_post = post.variance

        # calculcate the variance of the fantasy posteriors
        fantasy_posts = self.fantasy_models.posterior(
            (
                X.reshape(*X.shape[:-2], 1, *X.shape[-2:]).expand(
                    *X.shape[:-2], self.xs_exe.shape[0], *X.shape[-2:]
                )
            )
        )
        var_fantasy_posts = fantasy_posts.variance

        # calculate Shannon entropy for posterior given the current data
        h_current = 0.5 * torch.log(2 * torch.pi * var_post) + 0.5
        # sum the entropies from each independent posterior in the ModelList
        h_current_scalar = torch.sum(h_current, dim=-1)

        # calculate the Shannon entropy for the fantasy posteriors
        h_fantasies = 0.5 * torch.log(2 * torch.pi * var_fantasy_posts) + 0.5
        # sum the entropies from each independent posterior in the fantasy ModelList
        h_fantasies_scalar = torch.sum(h_fantasies, dim=-1)

        # compute the Monte-Carlo estimate of the Expected value of the entropy
        avg_h_fantasy = torch.mean(h_fantasies_scalar, dim=-2)

        # use the above entropies to compute the Expected Information Gain,
        # where the terms in the equation below correspond to the terms in
        # eq (4) of https://arxiv.org/pdf/2104.09460.pdf
        # (avg_h_fantasy is a Monte-Carlo estimate of the second term on the right)
        eig = h_current_scalar - avg_h_fantasy

        return eig.reshape(X.shape[:-2])
