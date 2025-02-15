#!/usr/bin/env python3

"""
Plot the extinction efficiency as a function of particle size.

This is a comparision of total extinction for non-absorbing and absorbing spheres.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython as mie

x = np.linspace(0.1, 100, 300)

plt.figure(figsize=(8,4.5))

qext, qsca, qback, g = mie.efficiencies_mx(1.5, x)
plt.plot(x, qext, color="red", label="1.5")

qext, qsca, qback, g = mie.efficiencies_mx(1.5 - 0.1j, x)
plt.plot(x, qext, color="blue", label="1.5-0.1j")

plt.title("Extinction by absorbing and non-absorbing spheres")
plt.xlabel(r"Size Parameter $x=\pi d/\lambda$    [—]")
plt.ylabel("Extinction Efficiency $Q_{ext}$    [—]")
plt.legend()
#plt.savefig("01.svg", format="svg")
#plt.show()
