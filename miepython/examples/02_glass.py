#!/usr/bin/env python3

"""
Plot the scattering efficiency for 4 micron glass spheres.

This graph shows scattering as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython as mie

radius = 2  # in microns
lambda0 = np.linspace(0.2, 1.2, 200)  # also in microns
x = 2 * np.pi * radius / lambda0

# from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
m2 = 1 + 1.03961212 / (1 - 0.00600069867 / lambda0**2)
m2 += 0.231792344 / (1 - 0.0200179144 / lambda0**2)
m2 += 1.01046945 / (1 - 103.560653 / lambda0**2)
m = np.sqrt(m2)

qext, qsca, qback, g = mie.mie(m, x)
plt.plot(lambda0 * 1000, qsca)

plt.title("BK7 glass spheres 4 micron diameter")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Efficiency (-)")
# plt.show()
