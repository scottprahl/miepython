miepython
==============
miepython is a Python module to calculate light scattering of solid spheres. Mie theory 
is used, following the procedure in given by Wiscombe 
<http://opensky.ucar.edu/islandora/object/technotes:232>

Usage
--------------
For examples and use cases, see test and doc folders.

### General Information and Comparisons
* [The basics](https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb) 
* [Scattering Efficiencies](https://github.com/scottprahl/miepython/blob/master/doc/02_efficiencies.ipynb) 
* [Scattering Phase Function](https://github.com/scottprahl/miepython/blob/master/doc/03_angular_scattering.ipynb) 
* [Rayleigh Scattering](https://github.com/scottprahl/miepython/blob/master/doc/04_rayleigh.ipynb) 
* [Simple Fog](https://github.com/scottprahl/miepython/blob/master/doc/05_fog.ipynb) 

### Simple Python Scripts
* [Extinction Efficiency of Absorbing and Non-Absorbing Spheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py) 
* [Four Micron Glass Spheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py) 
* [One Micron Water Droplets](https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py) 
* [Gold Nanospheres](https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py) 

### Algorithm Details
* [Generating Random Deviates](https://github.com/scottprahl/miepython/blob/master/doc/06_random_deviates.ipynb)
* [The Algorithm](https://github.com/scottprahl/miepython/blob/master/doc/07_algorithm.ipynb)

Installation via pip
--------------
   pip install miepython

Installation via github
--------------
Clone repository
   git clone https://github.com/scottprahl/miepython.git

Test by changing the miepython directory and doing
	nosetests miepython/test/test.py

Finally, add the miepython directory to PYTHONPATH 

To uninstall:
--------------
   pip uninstall miepython

Dependencies
--------------
For installation: setuptools

Required Python modules: numpy, matplotlib


License
--------------
miepython is licensed under the terms of the MIT license.