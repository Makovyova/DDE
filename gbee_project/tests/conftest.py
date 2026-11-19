"""Set process-wide test thread limits before collecting graph test modules.

These tests use tiny tensors: creating a large BLAS/OpenMP thread pool costs
far more than the numerical work.  Set environment limits before importing
NumPy, SciPy, or PyTorch, and keep them until this pytest process exits.
"""

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


def pytest_configure(config):
    """Also constrain an already imported Torch runtime before collection."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A third-party pytest plugin may already have initialized this pool.
        # Intra-op and BLAS limits above still apply; PyTorch forbids resetting
        # its inter-op pool after parallel work has started.
        pass
