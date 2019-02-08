# miepython

`miepython` is a Python module to calculate light scattering of solid spheres. Mie theory 
is used, following the [procedure in given by Wiscombe](http://opensky.ucar.edu/islandora/object/technotes:232)

## Usage

### Simple Example

```python
import miepython

m = 1.5  # index of refraction of sphere
x = 1.0  # dimensionless Mie size Parameter

qext, qsca, qback, g = miepython.mie(m,x)
print("The scattering efficiency  is %.3f" % qsca)
print("The backscatter efficiency is %.3f" % qback)
print("The scattering anisotropy  is %.3f" % g)
```

### General Overview of `miepython`
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

### Gory Algorithm Details
* [Generating Random Deviates](https://github.com/scottprahl/miepython/blob/master/doc/06_random_deviates.ipynb)
* [The Algorithm](https://github.com/scottprahl/miepython/blob/master/doc/07_algorithm.ipynb)

## Installation

    pip install miepython

## To uninstall:

    pip uninstall miepython

## License

`miepython` is licensed under the terms of the MIT license.