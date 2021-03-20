miepython
=========

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scottprahl/miepython/master?filepath=docs

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/miepython/blob/master

.. image:: https://img.shields.io/badge/nbviewer-docs-yellow.svg
   :target: https://nbviewer.jupyter.org/github/scottprahl/miepython/tree/master/docs

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://miepython.readthedocs.io

__________

`miepython` is a pure Python module to calculate light scattering by non-absorbing, partially-absorbing, or perfectly conducting spheres. Mie theory is used, following the procedure described by Wiscombe <http://opensky.ucar.edu/islandora/object/technotes:232>. This code has been validated against his results.

Extensive documentation is at <https://miepython.readthedocs.io>

This code provides functions for calculating the extinction efficiency, scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set of angles can be given to calculate the scattering for a sphere.

When comparing different Mie scattering codes, make sure that you're aware of the conventions used by each code.  `miepython` makes the following assumptions

#. the imaginary part of the complex index of refraction for absorbing spheres is *negative*.  

#. the scattering phase function is normalized so it equals the *single scattering albedo* when integrated over 4π steradians.

Installation
------------

Just use `pip`::

   pip install --user miepython

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

There are a few short scripts in the github repository.

* `Extinction Efficiency of Absorbing and Non-Absorbing Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py>`_ 
* `Four Micron Glass Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py>`_ 
* `One Micron Water Droplets <https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py>`_ 
* `Gold Nanospheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py>`_ 

License
-------

`miepython` is licensed under the terms of the MIT license.
