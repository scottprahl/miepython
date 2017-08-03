#!/usr/bin/env python3

"""
Plot the scattering cross section as a function of wavelength for 1 micron water droplets
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

num = 100
radius = 0.5                    # in microns
lam = np.linspace(0.2,1.2,num)  # also in microns
x = 2*np.pi*radius/lam

# from https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon-24.0C
m=(1+5.666959820E-1/(1-5.084151894E-3/lam**2)+1.731900098E-1/(1-1.818488474E-2/lam**2)+2.095951857E-2/(1-2.625439472E-2/lam**2)+1.125228406E-1/(1-1.073842352E1/lam**2))**.5

qqsca = np.zeros(num)

for i in range(num) :
    qext, qsca, qback, g = miepython.mie(m[i],x[i])
    qqsca[i]=qsca*np.pi*radius**2
    
plt.plot(lam*1000,qqsca)

plt.title(r"Water Droplets (1 $\mu$m diameter)")
plt.xlabel("Wavelength (nm)")
plt.ylabel(r"Scattering Cross Section ($\mu$m$^2$)")
plt.show()
