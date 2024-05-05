#!/usr/bin/env python3

"""
Plot the scattering efficiency for 4 micron glass spheres.

This graph shows scattering as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

def n_bk7(wavelength):
    """
    Refractive index of BK7 glass at a wavelength.
    
    Equation is from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT

    Args:
        wavelength: wavelength in microns
    Returns:
        index of refraction
    """
    m_squared = 1 + 1.03961212 / (1 - 0.00600069867 / lambda0**2)
    m_squared += 0.231792344 / (1 - 0.0200179144 / lambda0**2)
    m_squared += 1.01046945 / (1 - 103.560653 / lambda0**2)
    refractive_index = np.sqrt(m_squared)
    return refractive_index

radius = 2                      # in microns
lambda0 = np.linspace(0.2, 1.2, 1000)  # also in microns
x = 2 * np.pi * radius / lambda0
m = n_bk7(lambda0)

plt.figure(figsize=(8,4.5))
qext, qsca, qback, g = miepython.mie(m, x)
plt.plot(lambda0 * 1000, qsca)

plt.title("4µm diameter BK7 glass spheres")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Scattering Efficiency [–]")
#plt.savefig('../../docs/02_plot.svg')
plt.show()
