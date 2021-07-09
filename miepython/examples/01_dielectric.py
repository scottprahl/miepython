#!/usr/bin/env python3

"""
Plot the extinction efficiency as a function of particle size.

This is a comparision of total extinction for non-absorbing and absorbing
spheres.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

x = np.linspace(0.1, 100, 300)

# mie() will automatically try to do the right thing

qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, qext, color='red', label="1.5")

qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, qext, color='blue', label="1.5-0.1j")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qext")
plt.legend()
# plt.show()
