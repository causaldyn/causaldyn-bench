"""On a GPU, two defaults hold the suite to what it asserts on the CPU: without preallocation the
card is allocated as the tests need it rather than 75% of it at the first operation, and float32
matmuls run in float32 rather than TF32, whose 10-bit mantissa no tolerance here was set for."""

import os

# before jax is imported, so that no backend can have started without it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_default_matmul_precision", "highest")
