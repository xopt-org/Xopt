import traceback

from xopt.bayesian.utils import sampler_evaluate


class TestSampler:
    def test_sampler(self):
        ex = RuntimeError('this is an error')

        def f(x, *args):
            raise ex

        res = sampler_evaluate({'a': 'b'}, f)
        assert res['outputs'] == {
            "Exception": str(ex),
            "Traceback": traceback.print_tb(ex.__traceback__),
        }
