import pytest


class TestGeneratorOptions():
    def test_generator_options(self):
        from xopt.generator import GeneratorOptions
        opt = GeneratorOptions()
        assert opt.__config__.title is None
        assert opt.__config__.version is None

