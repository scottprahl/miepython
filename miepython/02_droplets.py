#!/usr/bin/env python3

"""
Plot the scattering efficiency as a function of wavelength for 4micron glass spheres
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

num = 100
radius = 2                      # in microns
lam = np.linspace(0.2,1.2,num)  # also in microns
x = 2*np.pi*radius/lam

# from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
m=np.sqrt(1+1.03961212/(1-0.00600069867/lam**2)+0.231792344/(1-0.0200179144/lam**2)+1.01046945/(1-103.560653/lam**2))

qqsca = np.zeros(num)

for i in range(num) :
    qext, qsca, qabs, qback, g = miepython.mie(m[i],x[i])
    qqsca[i]=qsca
    
plt.plot(lam*1000,qqsca)

plt.title("BK7 glass spheres 4 micron diameter")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Efficiency (-)")
plt.show()
