#!/usr/bin/env python3

"""
Plot the scattering cross section for 1 micron water droplets.

The plot shows the cross section as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

def n_water(wavelength):
    """
    Refractive index of water at wavelength.
    
    Equation is from https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon - 24.0C

    Args:
        wavelength: wavelength in microns
    Returns:
        index of refraction
    """
    m_squared = 1.0
    m_squared += 5.666959820E-1 / (1.0 - 5.084151894E-3 / wavelength**2)
    m_squared += 1.731900098E-1 / (1.0 - 1.818488474E-2 / wavelength**2)
    m_squared += 2.095951857E-2 / (1.0 - 2.625439472E-2 / wavelength**2)
    m_squared += 1.125228406E-1 / (1.0 - 1.073842352E1 / wavelength**2)
    refractive_index = np.sqrt(m_squared)
    return refractive_index

diameter = 1                               # microns
radius = diameter / 2                      # microns
num = 200                                  # points to plot
lambda_range = np.linspace(0.2, 1.2, num)
ref_index = n_water(lambda_range)
x = 2 * np.pi * radius / lambda_range

qext, qsca, qback, g = miepython.mie(ref_index, x)

plt.figure(figsize=(8,4.5))
plt.plot(lambda_range * 1000, qsca)
plt.title("%.2f µm Diameter Water Droplets" % diameter)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Scattering Cross Section [µm²]")
#plt.savefig('../../docs/03_plot.svg')
plt.show()
