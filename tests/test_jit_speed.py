"""
Benchmark test for miepython efficiency regression.

This test (contributed by Andrew Geiss) measures the runtime of the
`miepython.efficiencies_mx` function on N = 100,000 randomly generated
particles, and demonstrates the performance regression observed between
versions 2.5.5 and 3.0.0:

  • v2.5.5: 0.27 seconds for N=100000 jitted
  • v3.0.0: 4.41 seconds for N=100000 jitted
  • v3.0.1: 0.15 seconds for N=100000 jitted
  • v3.0.1: 4.00 seconds for N=100000 no jit

It toggles the Numba JIT backend via the `MIEPYTHON_USE_JIT` environment
variable, prints whether JIT is enabled, and reports the loaded miepython
version alongside the timing.

Original discussion and issue filed at:
https://github.com/scottprahl/miepython/issues/28
"""

import os
import numpy as np
from time import time

os.environ["MIEPYTHON_USE_JIT"] = "1"  # must come before importing miepython
import miepython as mie

# detect JIT for this benchmark
enabled = os.environ.setdefault("MIEPYTHON_USE_JIT", "1") == "1"

# Number of particles
N = 100_000

# Random refractive indices (n − ik) and size parameters x
refr = np.random.uniform(1.0, 2.0, N)
refi = np.exp(np.random.uniform(np.log(1e-4), np.log(1.0), N))
x = np.exp(np.random.uniform(np.log(0.01), np.log(100), N))
m = refr - 1j * refi

# Run the benchmark
t0 = time()
qext, qsca, qback, g = mie.efficiencies_mx(m, x)
print(f'JIT is {"enabled" if enabled else "not enabled"}, ' f"miepython version is {mie.__version__}")
print(f"{time() - t0:.3f} seconds when N={N}")
