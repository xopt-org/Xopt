import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from xopt.generators.bayesian.bax.algorithms import Algorithm


class ExpectedInformationGain(AnalyticAcquisitionFunction):
    r"""Single outcome expected information gain`
        Currently only works with model = SingleTaskGP (not ModelListGP)
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EIG = ExpectedInformationGain(model, algo, algo_params)
        >>> eig = EIG(test_X)
    """

    def __init__(self, model: Model, algo: type[Algorithm]) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
        """

        super().__init__(model=model)
        self.algo = algo

        # get sample-wise algorithm execution (BAX) results
        self.xs_exe, self.ys_exe, self.algo_results = self.algo.get_exe_paths(
            self.model
        )

        # Need to call the model on some data before we can condition_on_observations
        self.model(self.xs_exe[0, 0:1, :])

        # construct a batch of size n_samples fantasy models,
        # where each fantasy model is produced by taking the model
        # at the current iteration and conditioning it
        # on one of the sampled execution path subsequences:
        xs_exe_transformed = self.model.input_transform(self.xs_exe)
        ys_exe_transformed = self.model.outcome_transform(self.ys_exe)[0]
        self.fantasy_models = self.model.condition_on_observations(
            xs_exe_transformed, ys_exe_transformed
        )

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Information Gain on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Information Gain is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
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
        var_post = var_post.reshape(var_post.shape[:-1])

        # calculcate the variance of the fantasy posteriors
        fantasy_posts = self.fantasy_models.posterior(
            (
                X.reshape(*X.shape[:-2], 1, *X.shape[-2:]).expand(
                    *X.shape[:-2], self.xs_exe.shape[0], *X.shape[-2:]
                )
            )
        )
        var_fantasy_posts = fantasy_posts.variance
        var_fantasy_posts = var_fantasy_posts.reshape(var_fantasy_posts.shape[:-1])

        # calculate Shannon entropy for posterior given the current data
        h_current = 0.5 * torch.log(2 * torch.pi * var_post) + 0.5

        # calculate the Shannon entropy for the fantasy posteriors
        h_fantasies = 0.5 * torch.log(2 * torch.pi * var_fantasy_posts) + 0.5

        # compute the Monte-Carlo estimate of the Expected value of the entropy
        avg_h_fantasy = torch.mean(h_fantasies, dim=-2)

        # use the above entropies to compute the Expected Information Gain,
        # where the terms in the equation below correspond to the terms in
        # eq (4) of https://arxiv.org/pdf/2104.09460.pdf
        # (avg_h_fantasy is a Monte-Carlo estimate of the second term on the right)
        eig = h_current - avg_h_fantasy

        return eig.reshape(X.shape[:-2])
