miepython
=========

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: pypi

.. image:: https://img.shields.io/github/v/tag/scottprahl/miepython?label=github&color=68CA66
   :target: https://github.com/scottprahl/miepython
   :alt: github

.. image:: https://img.shields.io/conda/vn/conda-forge/miepython?label=conda&color=68CA66
   :target: https://github.com/conda-forge/miepython-feedstock
   :alt: conda

.. image:: https://zenodo.org/badge/99259684.svg
   :target: https://zenodo.org/badge/latestdoi/99259684
   :alt: doi
|
.. image:: https://img.shields.io/github/license/scottprahl/miepython?color=68CA66
   :target: https://github.com/scottprahl/miepython/blob/master/LICENSE.txt
   :alt: License

.. image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml
   :alt: Testing

.. image:: https://readthedocs.org/projects/miepython/badge?color=68CA66
   :target: https://miepython.readthedocs.io
   :alt: Docs

.. image:: https://img.shields.io/pypi/dm/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: Downloads

__________

``miepython`` is a pure Python module to calculate light scattering for
non-absorbing, partially-absorbing, or perfectly-conducting spheres. Mie
theory is used, following `the procedure described by Wiscombe
<http://opensky.ucar.edu/islandora/object/technotes:232>`_. This code has
been validated against his results. 


.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/mie-diagram1.png
   :alt: montage of laser images

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/mie-diagram2.png
   :alt: montage of laser images

This code provides functions for calculating the extinction efficiency, scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set of angles can be given to calculate the scattering for a sphere at each of those
angles.

Full documentation at <https://miepython.readthedocs.io>

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

Or `run this code in the cloud using Google Collaboratory <https://colab.research.google.com/github/scottprahl/miepython/blob/master>`_ by selecting the Jupyter notebook that interests you.

An example
----------

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

There are a few short scripts in the github repository.

* `Extinction Efficiency of Absorbing and Non-Absorbing Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py>`_
 
* `Four Micron Glass Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py>`_ 

* `One Micron Water Droplets <https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py>`_ 

* `Gold Nanospheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py>`_ 


License
-------

``miepython`` is licensed under the terms of the MIT license.
