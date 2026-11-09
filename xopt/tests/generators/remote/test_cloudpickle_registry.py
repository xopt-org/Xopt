"""Ensure both HTTP transports survive Xopt generator lookup and configuration."""

from xopt.generators import get_generator_dynamic
from xopt.generators.random import RandomGenerator
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator
from xopt.generators.remote.http import HTTPGenerator
from xopt.utils import get_generator_name
from gest_api.vocs import VOCS


def test_http_generator_registry_names_are_distinct():
    assert get_generator_dynamic("http") is HTTPGenerator
    assert get_generator_dynamic("cloudpickle_http") is CloudpickleHTTPGenerator
    assert HTTPGenerator.name == "http"
    assert CloudpickleHTTPGenerator.name == "cloudpickle_http"


def test_cloudpickle_generator_config_roundtrip():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    proxy = CloudpickleHTTPGenerator(
        base_url="http://127.0.0.1:8000",
        generator=RandomGenerator(vocs=vocs),
    )
    config = proxy.model_dump(mode="json")
    config["name"] = get_generator_name(proxy)
    config["generator"]["name"] = get_generator_name(proxy.generator)
    assert config.pop("name") == "cloudpickle_http"
    restored = get_generator_dynamic("cloudpickle_http").model_validate(config)
    assert isinstance(restored, CloudpickleHTTPGenerator)
    assert isinstance(restored.generator, RandomGenerator)
    assert restored.base_url == proxy.base_url
