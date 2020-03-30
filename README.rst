miepython
=========

`miepython` is a pure Python module to calculate light scattering by non-absorbing, partially-absorbing, or perfectly conducting spheres. Mie theory is used, following the procedure in given by Wiscombe <http://opensky.ucar.edu/islandora/object/technotes:232> and validated against his results.

This code provides functions for calculating the extinction efficiency, scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set of angles can be given to calculate the scattering for a sphere.

When comparing different Mie scattering codes, make sure that you're aware of the conventions used by each code.  `miepython` makes the following assumptions

#. the imaginary part of the complex index of refraction for absorbing spheres is *negative*.  

#. the scattering phase function is normalized so it equals the *single scattering albedo* when integrated over 4π steradians.

Installation
------------

Just use `pip`::

   pip install miepython

Usage
-----

The following code::

    import miepython
    
    m = 1.5-1j
    x = 1
    qext, qsca, qback, g = miepython.mie(m,x)

    print("The extinction efficiency  is %.3f" % qext)
    print("The scattering efficiency  is %.3f" % qsca)
    print("The backscatter efficiency is %.3f" % qback)
    print("The scattering anisotropy  is %.3f" % g)

should produce::

    The extinction efficiency  is 2.336
    The scattering efficiency  is 0.663
    The backscatter efficiency is 0.573
    The scattering anisotropy  is 0.192

Detailed documentation is available at <https://miepython.readthedocs.io>

There are a few short scripts in the github repository::

* `Extinction Efficiency of Absorbing and Non-Absorbing Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py>`_ 
* `Four Micron Glass Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py>`_ 
* `One Micron Water Droplets <https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py>`_ 
* `Gold Nanospheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py>`_ 

License
-------

`miepython` is licensed under the terms of the MIT license.
