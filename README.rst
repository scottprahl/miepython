.. |pypi| image:: https://img.shields.io/pypi/v/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: pypi

.. |github| image:: https://img.shields.io/github/v/tag/scottprahl/miepython?label=github&color=68CA66
   :target: https://github.com/scottprahl/miepython
   :alt: github

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/miepython?label=conda&color=68CA66
   :target: https://github.com/conda-forge/miepython-feedstock
   :alt: conda

.. |doi| image:: https://zenodo.org/badge/99259684.svg
   :target: https://zenodo.org/badge/latestdoi/99259684
   :alt: doi

.. |license| image:: https://img.shields.io/github/license/scottprahl/miepython?color=68CA66
   :target: https://github.com/scottprahl/miepython/blob/master/LICENSE.txt
   :alt: License

.. |test| image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml
   :alt: Testing

.. |docs| image:: https://readthedocs.org/projects/miepython/badge?color=68CA66
   :target: https://miepython.readthedocs.io
   :alt: Docs

.. |downloads| image:: https://img.shields.io/pypi/dm/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: Downloads

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: code style: black

miepython
=========

by Scott Prahl

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads| |black|

________

``miepython`` is a pure Python module to calculate light scattering for
non-absorbing, partially-absorbing, or perfectly-conducting spheres. Mie
theory is used, following `the procedure described by Wiscombe
<http://opensky.ucar.edu/islandora/object/technotes:232>`_. This code has
been validated against his results. 

This code provides functions for calculating the extinction efficiency,
scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set
of angles can be given to calculate the scattering for a sphere at each of those
angles.

Full documentation at <https://miepython.readthedocs.io>

Version 3 changes (in progress)
--------------------------------

This version contains major changes to the code base and **has API breaking changes**.
If you don't need the new functionality for fields, then you can continue to use the
last version with the old API: 2.5.5

Version 3.0 has many changes, but the major changes are:

    * a complete overhaul of API
    * added support to calculate Mie coefficients for fields inside sphere
    * added support for calculating electric and magnetic fields

Pay Attention!
--------------

When comparing different Mie scattering codes, make sure that you're aware of the conventions used by each code.  ``miepython`` makes the following assumptions

#. the imaginary part of the complex index of refraction for absorbing spheres is *negative*.  

#. the scattering phase function is normalized so it equals the *single scattering albedo* when integrated over 4π steradians.  As of version 2.3, this can be changed.

Installation
---------------

Use ``pip``::

    pip install miepython

or ``conda``::

    conda install -c conda-forge miepython

An example
----------

The following code::

    import miepython as mie
    
    m = 1.5 - 1j      # refractive index of sphere
    d = 100           # nm diameter of sphere
    lambda0 = 314.15  # nm wavelength in vacuum

    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0)

    print("The extinction efficiency  is %.3f" % qext)
    print("The scattering efficiency  is %.3f" % qsca)
    print("The backscatter efficiency is %.3f" % qback)
    print("The scattering anisotropy  is %.3f" % g)

should produce::

    The extinction efficiency  is 2.336
    The scattering efficiency  is 0.663
    The backscatter efficiency is 0.573
    The scattering anisotropy  is 0.192

There are a few short python scripts in the github repository.

* `Extinction Efficiency of Absorbing and Non-Absorbing Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py>`_

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/01.svg
   :alt: Absorbing and non-absorbing spheres
 
* `Four Micron Glass Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py>`_ 

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/02.svg
   :alt: Glass spheres with resonance spike

* `One Micron Water Droplets <https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py>`_ 

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/03.svg
   :alt: Water Droplets

* `Gold Nanospheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py>`_ 

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/04.svg
   :alt: Gold nanospheres

License
-------

``miepython`` is licensed under the terms of the MIT license.

