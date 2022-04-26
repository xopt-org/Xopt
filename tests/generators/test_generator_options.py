import pytest


class TestGeneratorOptions():
    def test_generator_options(self):
        from xopt.generators.options import GeneratorOptions
        from pydantic import BaseModel
        opt = GeneratorOptions()

        # try to construt a generator object with duplicate keys
        class MySubGenerator(GeneratorOptions):
            name: str = "my_generator"

        class MyGenerator(GeneratorOptions):
            gen: MySubGenerator = MySubGenerator()

        opt = MySubGenerator()
        with pytest.raises(RuntimeError):
            MyGenerator(name="my_generator")
