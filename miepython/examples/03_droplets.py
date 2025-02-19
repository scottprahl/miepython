#!/usr/bin/env python3

"""
Plot the scattering cross section for 1 micron water droplets.

The plot shows the cross section as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython as mie

radius = 0.5  # in microns
lambda0 = np.linspace(0.2, 1.2, 100)  # also in microns
geometric_cross_section = np.pi * radius**2

# from https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon - 24.0C
m2 = 1.0
m2 += 5.666959820e-1 / (1.0 - 5.084151894e-3 / lambda0**2)
m2 += 1.731900098e-1 / (1.0 - 1.818488474e-2 / lambda0**2)
m2 += 2.095951857e-2 / (1.0 - 2.625439472e-2 / lambda0**2)
m2 += 1.125228406e-1 / (1.0 - 1.073842352e1 / lambda0**2)
m = np.sqrt(m2)

plt.figure(figsize=(8, 4.5))

qext, qsca, qback, g = mie.efficiencies(m, 2 * radius, lambda0)
sigma_sca = qsca * geometric_cross_section

plt.plot(lambda0 * 1000, sigma_sca)
plt.title("Water Droplets (1 µm diameter)")
plt.xlabel("Wavelength    [nm]")
plt.ylabel("Scattering Cross Section    [µm²]")
# plt.savefig("03.svg", format="svg")
# plt.show()
