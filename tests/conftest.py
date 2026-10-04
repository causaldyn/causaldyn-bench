"""On a GPU, two defaults hold the suite to what it asserts on the CPU: without preallocation the
card is allocated as the tests need it rather than 75% of it at the first operation, and float32
matmuls run in float32 rather than TF32, whose 10-bit mantissa no tolerance here was set for."""

import os

# before jax is imported, so that no backend can have started without it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="session")
def other_kernels() -> str | None:
    """Why numpy's float64 exp, log and tanh here are not the kernels the committed worlds and
    records were made with, or None when they are. They were made on x86 without AVX-512, where
    the baseline and AVX2 kernels return the same bits; the AVX-512 kernels are other code, so
    a test that reads a drawn world's exact bits holds only where this is None."""
    try:
        from numpy.lib.introspect import opt_func_info
    except ImportError:  # numpy < 2 does not name the kernel it runs
        return f"numpy {np.__version__} does not name its kernels"
    found = opt_func_info(func_name="^(exp|expm1|log|log1p|tanh)$", signature="float64")
    running = {name: kernels["dd"]["current"] for name, kernels in found.items()}
    if all(k == "X86_V3" or k.startswith("baseline(X86") for k in running.values()):
        return None
    return f"numpy's float64 kernels here are {running}, not the exporting machine's"
