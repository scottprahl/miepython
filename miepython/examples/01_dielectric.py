#!/usr/bin/env python3

"""Plot the extinction efficiency as a function of particle size."""

import numpy as np
import matplotlib.pyplot as plt
import miepython

x = np.linspace(0.1, 100, 1000)

plt.figure(figsize=(8, 4.5))
qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, qext, color='red', label="$n=1.5$")

qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, qext, color='blue', label="$n=1.5-0.1j$")

plt.title("Absorbing and Non-absorbing Spheres")
plt.xlabel("Size Parameter [–]")
plt.ylabel("Extinction Efficiency $Q_{ext}$ [–]")
plt.legend()
# plt.savefig('../../docs/01_plot.svg')
plt.show()
