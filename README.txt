miepython
==============
miepython is a Python module to calculate scattering and absorption properties of solid spheres and core-shell structures. Mie theory is used, following the procedure in given by Wiscombe.


Images
--------------
In the image below, scattering intensity is shown for a 100 nm radius dielectric sphere with refractive index n = 3.7. 

<p align="center">
  <img src="images/sphere_scattering.png?raw=true" width="600">
</p>


Usage
--------------

For examples and use cases, see examples folder

For full documentation, see docs folder


Installation
--------------
First, clone (or download) this repository and cd into it:

```shell
git clone https://github.com/johnaparker/MiePy && cd MiePy
```

Then, install MiePy via pip (or use just setuptools):

```shell
pip install .                    #using pip
python setup.py install          #using just setuptools
```

To uninstall:

```shell
pip uninstall miepy 
```

To use miepython without installation, add the miepython directory to your PYTHONPATH


Dependencies
--------------
For installation: setuptools

Required Python modules: numpy, matplotlib


License
--------------
miepython is licensed under the terms of the MIT license.