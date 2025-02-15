#!/usr/bin/env python3

"""
Plot the scattering cross section for 1 micron water droplets.

The plot shows the cross section as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython as mie

num = 100
radius = 0.5  # in microns
lambda0 = np.linspace(0.2, 1.2, num)  # also in microns
x = 2 * np.pi * radius / lambda0

# from https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon - 24.0C
m2 = 1.0
m2 += 5.666959820e-1 / (1.0 - 5.084151894e-3 / lambda0**2)
m2 += 1.731900098e-1 / (1.0 - 1.818488474e-2 / lambda0**2)
m2 += 2.095951857e-2 / (1.0 - 2.625439472e-2 / lambda0**2)
m2 += 1.125228406e-1 / (1.0 - 1.073842352e1 / lambda0**2)
m = np.sqrt(m2)

qext, qsca, qback, g = mie.efficiencies_mx(m, x)

plt.plot(lambda0 * 1000, qsca)
plt.title("Water Droplets (1 µm diameter)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Cross Section (µm²)")
# plt.show()
