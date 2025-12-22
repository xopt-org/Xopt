from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from xopt import Evaluator, VOCS, Xopt
from xopt.vocs import select_best, get_variable_data
from xopt.errors import SeqGeneratorError
from xopt.generators.sequential.rcds import RCDSGenerator
from xopt.generators.sequential import rcds
from xopt.resources.testing import TEST_VOCS_BASE


def f_RCDS_minimize(input_dict):
    p = []
    for i in range(2):
        p.append(input_dict[f"p{i}"])

    obj = np.linalg.norm(p)
    outcome_dict = {"f": obj}

    return outcome_dict


def eval_f_linear_pos(x):
    return {"y1": np.sum([x**2 for x in x.values()])}


def eval_f_linear_neg(x):
    return {"y1": -np.sum([x**2 for x in x.values()])}


def eval_f_linear_offset(x):  # offset the optimal solution
    return {"y1": np.sum([(x - 2) ** 2 for x in x.values()])}


class TestRCDSGenerator:
    def test_rcds_generate_multiple_points(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}

        gen = RCDSGenerator(vocs=test_vocs)

        # Try to generate multiple samples
        with pytest.raises(SeqGeneratorError):
            gen.generate(2)

    def test_rcds_options(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}

        gen = RCDSGenerator(vocs=test_vocs)

        with pytest.raises(ValidationError):
            gen.step = 0

        with pytest.raises(ValidationError):
            gen.tol = 0

    def test_rcds_yaml(self):
        YAML = """
        stopping_condition:
            name: MaxEvaluationsCondition
            max_evaluations: 100
        generator:
            name: rcds
            init_mat: null
            noise: 0.00001
            step: 0.01
        evaluator:
            function: xopt.resources.test_functions.tnk.evaluate_TNK
        vocs:
            variables:
                x1: [0, 1]
                x2: [0, 1]
            objectives:
                y1: MINIMIZE
        """
        X = Xopt.from_yaml(YAML)
        X.random_evaluate(1)

        # test running multiple steps
        for i in range(10):
            X.step()

        assert X.generator.is_active
        assert X.generator._last_candidate is not None

        X.generator.reset()
        assert not X.generator.is_active

    @pytest.mark.parametrize(
        "fun, obj, x_opt, max_iter",
        [
            (eval_f_linear_pos, "MINIMIZE", np.zeros(10), 300),
            (eval_f_linear_neg, "MAXIMIZE", np.zeros(10), 300),
            (eval_f_linear_offset, "MINIMIZE", 2 * np.ones(10), 300),
        ],
    )
    def test_rcds_convergence(self, fun, obj, x_opt, max_iter):
        variables = {f"x{i}": [-5, 5] for i in range(len(x_opt))}
        objectives = {"y1": obj}
        vocs = VOCS(variables=variables, objectives=objectives)
        generator = RCDSGenerator(step=0.01, noise=0.00001, vocs=vocs)
        evaluator = Evaluator(function=fun)
        X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

        if x_opt.sum():  # if the optimal solution is not 0
            X.evaluate_data({f"x{i}": 1.2 for i in range(len(x_opt))})
        else:
            X.random_evaluate(1)
        for i in range(max_iter):
            X.step()

        idx, best, _ = select_best(X.vocs, X.data)
        xbest = get_variable_data(X.vocs, X.data.loc[idx, :]).to_numpy().flatten()
        if obj == "MINIMIZE":
            assert best[0] >= 0.0
            assert best[0] <= 0.001
        else:
            assert best[0] <= 0.0
            assert best[0] >= -0.001
        assert np.allclose(xbest, x_opt, rtol=0, atol=1e-1)

    def test_bracketmin_exceptions(self):
        sm = rcds.BracketMinStateMachine(0.1, np.zeros(2), 1.0, np.ones(2), 0.1)
        sm.phase = "finished"
        with pytest.raises(rcds.StateMachineFinishedError):
            sm.propose()
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.propose()
        sm.phase = "forward_first"
        sm.pending = False
        sm.current_branch = "invalid"
        sm.pending = True
        with pytest.raises(Exception):
            sm.update_obj(1.0)
        sm.pending = False
        with pytest.raises(Exception):
            sm.update_obj(1.0)

    def test_linescan_exceptions(self):
        sm = rcds.LineScanStateMachine(
            np.zeros(2), 1.0, np.ones(2), 0, 1, 6, np.zeros((0, 2))
        )
        sm.phase = "finished"
        with pytest.raises(rcds.StateMachineFinishedError):
            sm.propose()
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.propose()
        sm.phase = "linescan_loop"
        sm.pending = False
        sm.current_branch = "invalid"
        sm.pending = True
        with pytest.raises(Exception):
            sm.update_obj(1.0)
        sm.pending = False
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.update_obj(1.0)

    def test_getminalongdirparab_exceptions(self):
        sm = rcds.GetMinAlongDirParabStateMachine(np.zeros(2), 1.0, np.ones(2))
        sm.phase = "finished"
        with pytest.raises(rcds.StateMachineFinishedError):
            sm.propose()
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.propose()
        sm.pending = True
        with pytest.raises(Exception):
            sm.propose()
        sm.pending = False
        sm.phase = "bracketmin"
        sm.bm.pending = False
        with pytest.raises(Exception):
            sm.update_obj(1.0)
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.update_obj(1.0)

    def test_powell_exceptions(self):
        sm = rcds.PowellMainStateMachine(np.zeros(2), 0.1)
        sm.phase = "finished"
        with pytest.raises(rcds.StateMachineFinishedError):
            sm.propose()
        sm.phase = "init"
        sm.pending = True
        with pytest.raises(Exception):
            sm.propose()
        sm.pending = False
        sm.phase = "invalid"
        with pytest.raises(Exception):
            sm.propose()
        sm.phase = "init_wait"
        sm.pending = False
        with pytest.raises(Exception):
            sm.update_obj(1.0)
        sm.phase = "invalid"
        sm.pending = True
        with pytest.raises(Exception):
            sm.update_obj(1.0)

    def test_rcds_add_data_and_set_data(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        gen = rcds.RCDSGenerator(vocs=test_vocs)
        # _powell is None, should not fail
        gen._powell = None
        df = pd.DataFrame({"y1": [1.0]})
        gen._add_data(df)
        # _set_data sets data
        gen._set_data(df)
        assert gen.data is df

    def test_rcds_generate_out_of_bounds(self):
        # This will force the candidate to be out of bounds and test the while loop
        class DummyPowell:
            def __init__(self):
                self.calls = 0

            def propose(self):
                self.calls += 1
                if self.calls == 1:
                    return np.array([2.0, 2.0])  # out of bounds
                return np.array([0.5, 0.5])

            def update_obj(self, val):
                self.last = val

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.constraints = {}
        gen = rcds.RCDSGenerator(vocs=test_vocs)
        gen._powell = DummyPowell()
        gen.vocs.variables = {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}
        result = gen._generate()
        assert isinstance(result, list)
        assert all(0 <= v <= 1 for v in result[0].values())

    def test_powell_direction_update_gmadp_finished(self):
        # Test the highlighted code: when current_gmadp.propose() raises StateMachineFinishedError
        class DummyGMADP:
            def __init__(self):
                self.called = False

            def propose(self):
                if not self.called:
                    self.called = True
                    raise rcds.StateMachineFinishedError((np.array([1.0, 2.0]), 3.0, 4))
                return np.array([0.5, 0.5])

        sm = rcds.PowellMainStateMachine(np.zeros(2), 0.1)
        sm.phase = "direction_update"
        sm.current_gmadp = DummyGMADP()
        sm.x_current = np.array([0.0, 0.0])
        sm.f_current = 0.0
        sm.nf = 0
        sm.Dmat = np.identity(2)
        # Should handle StateMachineFinishedError and continue to iteration_end, then to line_search
        result = sm.propose()
        assert sm.phase == "line_search"
        assert sm.nf == 4
        assert np.all(sm.x_current == np.array([1.0, 2.0]))
        assert sm.f_current == 3.0
        assert isinstance(result, np.ndarray)

    def test_powell_update_obj_direction_update(self):
        # Test update_obj in 'direction_update' phase delegates to current_gmadp.update_obj
        class DummyGMADP:
            def __init__(self):
                self.updated = False

            def update_obj(self, val):
                self.updated = val

        sm = rcds.PowellMainStateMachine(np.zeros(2), 0.1)
        sm.phase = "direction_update"
        sm.pending = True
        sm.current_gmadp = DummyGMADP()
        sm.update_obj(42)
        assert sm.current_gmadp.updated == 42
