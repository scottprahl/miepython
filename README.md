miepython
==============
miepython is a Python module to calculate light scattering of solid spheres. Mie theory is used, following the procedure in given by Wiscombe http://opensky.ucar.edu/islandora/object/technotes:232


Usage
--------------
For examples and use cases, see test folder

Installation
--------------
First, clone (or download) this repository and cd into it:

```shell
git clone https://github.com/scottprahl/miepython.git
cd miepython
```

Then, install miepython via pip (or use just setuptools):

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