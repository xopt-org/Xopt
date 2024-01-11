from typing import Optional

from pydantic import Field
from scipy.stats import qmc
from typing_extensions import Annotated

from xopt.generator import Generator


class LatinHypercubeGenerator(Generator):
    """
    Latin hypercube sampling (intended for sampling data for surrogate model generation). Samples get generated in
    batches of size batch_size and then returned to user in chunks of requested size.
    """

    name = "latin_hypercube"
    supports_batch_generation: bool = True
    supports_multi_objective: bool = True
    batch_size: Annotated[
        Optional[int],
        Field(
            4096,
            strict=True,
            ge=1,
            description="How many samples to \
        generate at one time.",
        ),
    ]
    scramble: Annotated[
        Optional[bool],
        Field(
            True,
            description="When False, center samples within cells of a \
        multi-dimensional grid. Otherwise, samples are randomly placed within cells of the grid. See scipy \
        documentation.",
        ),
    ]
    optimization: Annotated[
        Optional[str],
        Field(
            "random-cd",
            description="Whether to use an optimization \
        scheme to improve the quality after sampling. See scipy documentation.",
        ),
    ]
    strength: Annotated[
        Optional[int],
        Field(
            1,
            strict=True,
            ge=1,
            le=2,
            description="Strength of the LHS. \
        strength=1 produces a plain LHS while strength=2 produces an orthogonal array based LHS of strength 2. \
        See scipy documentation.",
        ),
    ]
    seed: Annotated[
        Optional[int], Field(None, description="Random seed. See scipy documentation.")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sampler = qmc.LatinHypercube(
            d=len(self.vocs.variables),
            scramble=self.scramble,
            optimization=self.optimization,
            strength=self.strength,
            seed=self.seed,
        )
        self._samples = []

    def initialize_batch(self):
        names = self.vocs.variable_names
        rows = qmc.scale(
            self._sampler.random(n=self.batch_size),
            [self.vocs.variables[k][0] for k in names],
            [self.vocs.variables[k][1] for k in names],
        )
        rows = [{name: ele for name, ele in zip(names, row)} for row in rows]
        self._samples = [{**row, **self.vocs.constants} for row in rows]

    def generate(self, n_candidate) -> list[dict]:
        ret = []
        while len(ret) < n_candidate:
            n_needed = n_candidate - len(ret)
            if n_needed < len(self._samples):
                ret.extend(self._samples[:n_needed])
                self._samples = self._samples[n_needed:]
            else:
                ret.extend(self._samples)
                self.initialize_batch()
        return ret
