# miepython

`miepython` is a pure Python module to calculate light scattering by non-absorbing, partially-absorbing, or perfectly conducting spheres. Mie theory 
is used, following the [procedure in given by Wiscombe](http://opensky.ucar.edu/islandora/object/technotes:232)

This code provides functions for calculating the extinction efficiency, scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set of angles can be given and to calculate the scattering for a sphere.

## Usage

### Simple Example

```python
import miepython

m = 1.5  # index of refraction of sphere
x = 1.0  # dimensionless Mie size parameter

qext, qsca, qback, g = miepython.mie(m,x)
print("The scattering efficiency  is %.3f" % qsca)
print("The backscatter efficiency is %.3f" % qback)
print("The scattering anisotropy  is %.3f" % g)
```

> The scattering efficiency  is 0.215

> The backscatter efficiency is 0.187

> The scattering anisotropy  is 0.199


### Detailed Documentation
* [Mie Size Parameter, Complex Index of Refraction](https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb) 
* [Cross Sections and Efficiencies](https://github.com/scottprahl/miepython/blob/master/doc/02_efficiencies.ipynb) 
* [Scattering Phase Function](https://github.com/scottprahl/miepython/blob/master/doc/03_angular_scattering.ipynb) 
* [Rayleigh Scattering](https://github.com/scottprahl/miepython/blob/master/doc/04_rayleigh.ipynb) 
* [Simple Fog](https://github.com/scottprahl/miepython/blob/master/doc/05_fog.ipynb) 
* [Large Spheres](https://github.com/scottprahl/miepython/blob/master/doc/08-large_spheres.ipynb)

### Small Scripts Using `miepython`
* [Extinction Efficiency of Absorbing and Non-Absorbing Spheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py) 
* [Four Micron Glass Spheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py) 
* [One Micron Water Droplets](https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py) 
* [Gold Nanospheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py) 

### Gory Details
* [Generating Random Deviates for Monte Carlo](https://github.com/scottprahl/miepython/blob/master/doc/06_random_deviates.ipynb)
* [The Algorithm](https://github.com/scottprahl/miepython/blob/master/doc/07_algorithm.ipynb)

## Installation

    pip install miepython

## To uninstall:

    pip uninstall miepython

## License

`miepython` is licensed under the terms of the MIT license.