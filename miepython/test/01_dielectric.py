#!/usr/bin/env python3

"""
Plot the extinction efficiency as a function of particle size
for non-absorbing and absorbing spheres
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython

num = 100

x = np.linspace(0.1,100,num)
qqext1 = np.zeros(num)
qqext2 = np.zeros(num)

for i in range(num) :
    qext, qsca, qabs, qback, g = miepython.mie(1.5,x[i])
    qqext1[i]=qext
    qext, qsca, qabs, qback, g = miepython.mie(1.5-0.1j,x[i])
    qqext2[i]=qext
    
plt.plot(x,qqext1)
plt.plot(x,qqext2)

plt.xlabel("Size Parameter (-)")
plt.ylabel("Qext")
plt.show()
