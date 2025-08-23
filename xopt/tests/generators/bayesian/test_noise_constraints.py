import logging
from copy import deepcopy

import pytest
import torch

from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.custom_botorch.noise_constrained_acq import (
    UCqExpectedImprovement,
)
from xopt.resources.testing import (
    TEST_VOCS_BASE,
    TEST_VOCS_DATA,
    check_generator_tensor_locations,
    create_set_options_helper,
    generate_without_warnings,
)

set_options = create_set_options_helper(data=TEST_VOCS_DATA)
device_map = {False: torch.device("cpu"), True: torch.device("cuda:0")}

logger = logging.getLogger(__name__)

use_cuda = False


@pytest.mark.parametrize("turbo", [None])
@pytest.mark.parametrize("log", [False])
def test_ei_nc(log, turbo):
    # logger.info(f"Running test for {problem_cfg} with log {log} and turbo {turbo}")

    vocs = deepcopy(TEST_VOCS_BASE)
    vocs.constraints = {}
    gen = ExpectedImprovementGenerator(
        vocs=vocs,
        variance_limits={vocs.objective_names[0]: 0.1},
        variance_eta={vocs.objective_names[0]: 0.001},
    )
    set_options(gen, use_cuda=False, add_data=True)

    candidate = generate_without_warnings(gen, 1)
    assert len(candidate) == 1

    candidate = generate_without_warnings(gen, 2)
    assert len(candidate) == 2

    check_generator_tensor_locations(gen, device_map[use_cuda])

    assert isinstance(gen.get_acquisition(gen.model), UCqExpectedImprovement)
