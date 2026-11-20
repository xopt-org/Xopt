"""Internal CUDA initialization workaround for Bayesian optimization."""

import logging
import threading

import torch


logger = logging.getLogger(__name__)

_cuda_initialization_lock = threading.Lock()
_initialized_cuda_devices: set[torch.device] = set()


def ensure_cuda_linalg_initialized(device: torch.device | str) -> None:
    """Safely initialize CUDA linear algebra on first use.

    Workaround for PyTorch issue #90613:
    https://github.com/pytorch/pytorch/issues/90613

    Concurrent first-use CUDA linear algebra operations may fail
    while PyTorch lazily loads its CUDA linear algebra library,
    producing "lazy wrapper should be called at most once".

    Independently reproduced with PyTorch 2.15.0+cu132 using
    concurrent calls to torch.linalg.cholesky_ex().

    This workaround applies to both local and remote Bayesian
    Generators. No additional user configuration is required.

    Remove when a supported PyTorch release resolves the issue
    and passes cold-start concurrency regression tests.
    """
    device = torch.device(device)

    # CPU workloads must never trigger CUDA initialization,
    # even when a GPU is available on the host.
    if device.type != "cuda":
        return

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    # Serialize only first-use initialization. Never hold this
    # lock during GP model fitting or acquisition optimization.
    with _cuda_initialization_lock:
        if device in _initialized_cuda_devices:
            return

        with torch.no_grad():
            # A small positive-definite matrix triggers PyTorch's
            # lazy CUDA linear algebra initialization.
            matrix = torch.eye(32, dtype=torch.float64, device=device)
            _, info = torch.linalg.cholesky_ex(matrix)

            if info.item() != 0:
                raise RuntimeError("CUDA linear algebra initialization failed")

            torch.cuda.synchronize(device)

        # Record success only after initialization completes.
        _initialized_cuda_devices.add(device)

        logger.info("CUDA linear algebra initialized on %s", device)
