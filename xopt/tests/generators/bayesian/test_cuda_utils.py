"""Unit tests for automatic CUDA linear algebra initialization.

CUDA operations are mocked so these tests also run on CPU-only
CI workers. Real GPU behavior is covered by concurrency tests.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

from xopt.generators.bayesian import cuda_utils


@pytest.fixture
def cuda_backend(monkeypatch):
    """Provide isolated initialization state and mocked CUDA operations."""
    initialized = set()

    monkeypatch.setattr(
        cuda_utils,
        "_initialized_cuda_devices",
        initialized,
    )

    matrix = Mock(name="matrix")
    info = Mock(name="cholesky_info")
    info.item.return_value = 0

    eye = Mock(return_value=matrix)
    cholesky = Mock(return_value=(Mock(), info))
    synchronize = Mock()
    current_device = Mock(return_value=0)

    monkeypatch.setattr(torch, "eye", eye)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", cholesky)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)

    operations = MagicMock()
    operations.attach_mock(eye, "eye")
    operations.attach_mock(cholesky, "cholesky")
    operations.attach_mock(synchronize, "synchronize")

    return {
        "initialized": initialized,
        "matrix": matrix,
        "info": info,
        "eye": eye,
        "cholesky": cholesky,
        "synchronize": synchronize,
        "current_device": current_device,
        "operations": operations,
    }


def test_cpu_device_bypasses_cuda(cuda_backend):
    cuda_utils.ensure_cuda_linalg_initialized("cpu")
    cuda_utils.ensure_cuda_linalg_initialized(torch.device("cpu"))

    assert not cuda_backend["initialized"]
    assert cuda_backend["operations"].mock_calls == []
    cuda_backend["current_device"].assert_not_called()


def test_first_cuda_initialization(cuda_backend):
    device = torch.device("cuda:0")

    cuda_utils.ensure_cuda_linalg_initialized(device)

    assert cuda_backend["operations"].mock_calls == [
        call.eye(32, dtype=torch.float64, device=device),
        call.cholesky(cuda_backend["matrix"]),
        call.synchronize(device),
    ]

    assert cuda_backend["initialized"] == {device}


def test_repeated_requests_initialize_once(cuda_backend):
    for _ in range(10):
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert cuda_backend["eye"].call_count == 1
    assert cuda_backend["cholesky"].call_count == 1
    assert cuda_backend["synchronize"].call_count == 1


def test_concurrent_requests_initialize_once(cuda_backend):
    workers = 8
    barrier = Barrier(workers)

    def initialize():
        barrier.wait(timeout=30)
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(initialize) for _ in range(workers)]

        for future in futures:
            future.result(timeout=30)

    assert cuda_backend["eye"].call_count == 1
    assert cuda_backend["cholesky"].call_count == 1
    assert cuda_backend["synchronize"].call_count == 1

    assert cuda_backend["initialized"] == {torch.device("cuda:0")}


def test_cholesky_failure_is_not_cached(cuda_backend):
    cuda_backend["info"].item.return_value = 1

    with pytest.raises(
        RuntimeError,
        match="CUDA linear algebra initialization failed",
    ):
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert not cuda_backend["initialized"]
    cuda_backend["synchronize"].assert_not_called()


def test_retry_after_cholesky_failure(cuda_backend):
    cuda_backend["info"].item.side_effect = [1, 0]

    with pytest.raises(RuntimeError):
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert not cuda_backend["initialized"]

    cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert cuda_backend["cholesky"].call_count == 2
    assert cuda_backend["synchronize"].call_count == 1

    assert cuda_backend["initialized"] == {torch.device("cuda:0")}


def test_synchronization_failure_is_not_cached(cuda_backend):
    cuda_backend["synchronize"].side_effect = [
        RuntimeError("CUDA synchronization failed"),
        None,
    ]

    with pytest.raises(
        RuntimeError,
        match="CUDA synchronization failed",
    ):
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert not cuda_backend["initialized"]

    cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert cuda_backend["cholesky"].call_count == 2
    assert cuda_backend["synchronize"].call_count == 2

    assert cuda_backend["initialized"] == {torch.device("cuda:0")}


def test_multiple_cuda_devices(cuda_backend):
    for device in ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]:
        cuda_utils.ensure_cuda_linalg_initialized(device)

    assert cuda_backend["initialized"] == {
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    }

    assert cuda_backend["synchronize"].call_args_list == [
        call(torch.device("cuda:0")),
        call(torch.device("cuda:1")),
    ]

    assert cuda_backend["cholesky"].call_count == 2


def test_implicit_cuda_device_index(cuda_backend):
    cuda_backend["current_device"].return_value = 1

    cuda_utils.ensure_cuda_linalg_initialized("cuda")
    cuda_utils.ensure_cuda_linalg_initialized("cuda:1")

    cuda_backend["current_device"].assert_called_once()

    assert cuda_backend["initialized"] == {torch.device("cuda:1")}

    assert cuda_backend["cholesky"].call_count == 1


def test_pytorch_exception_allows_retry(cuda_backend):
    cuda_backend["cholesky"].side_effect = RuntimeError("PyTorch operation failed")

    with pytest.raises(
        RuntimeError,
        match="PyTorch operation failed",
    ):
        cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert not cuda_backend["initialized"]

    cuda_backend["cholesky"].side_effect = None

    cuda_utils.ensure_cuda_linalg_initialized("cuda:0")

    assert cuda_backend["initialized"] == {torch.device("cuda:0")}

    assert cuda_backend["cholesky"].call_count == 2
