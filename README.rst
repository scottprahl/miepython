miepython
=========

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/miepython.svg
   :target: https://pypi.org/project/miepython/

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/miepython/blob/master

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://miepython.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green.svg
   :target: https://github.com/scottprahl/miepython

.. image:: https://img.shields.io/badge/MIT-license-yellow.svg
   :target: https://github.com/scottprahl/miepython/blob/master/LICENSE.txt

.. image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml

__________

``miepython`` is a pure Python module to calculate light scattering for
non-absorbing, partially-absorbing, or perfectly-conducting spheres. Mie
theory is used, following `the procedure described by Wiscombe
<http://opensky.ucar.edu/islandora/object/technotes:232>`_. This code has
been validated against his results. 

This code provides functions for calculating the extinction efficiency, scattering efficiency, backscattering, and scattering asymmetry. Moreover, a set of angles can be given to calculate the scattering for a sphere at each of those
angles.

Full documentation at <https://miepython.readthedocs.io>

Pay Attention!
--------------

When comparing different Mie scattering codes, make sure that you're aware of the conventions used by each code.  ``miepython`` makes the following assumptions

#. the imaginary part of the complex index of refraction for absorbing spheres is *negative*.  

#. the scattering phase function is normalized so it equals the *single scattering albedo* when integrated over 4π steradians.

## Normalization of the scattered light

Mie scattering is used in a wide variety of disciplines to describe the scattering pattern from spheres.  Not surprisingly a bunch of different normalizations have arisen that can be described based on the integrating the scattering function over all directions ($4\pi$ steradians).

Integrating the scattering phase function all solid angles suggests the following dimensionless possibilities which have been chosen by one or more authors for their convenience.  `miepython` now supports all of these.

.. math::

        \int_{4\pi} p(\theta,\phi) \,d\Omega = a       \qquad\qquad \mbox{albedo (default)}

.. math::

    \begin{align}
    \int_{4\pi} p(\theta,\phi) \,d\Omega = a       \qquad&\qquad \mbox{albedo (default)}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega = 1       \qquad&\qquad \mbox{one}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega = 4\pi    \qquad&\qquad \mbox{4pi}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega = Q_{sca} \qquad&\qquad \mbox{qsca}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega = Q_{ext} \qquad&\qquad \mbox{qext}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega = 4\pi x^2 Q_{sca}\qquad&\qquad \mbox{Bohren}\\[2mm]
    \int_{4\pi} p(\theta,\phi) \,d\Omega =\pi x^2 Q_{sca}\qquad&\qquad \mbox{Wiscombe}\\[2mm]
    \end{align}


where $x=2\pi a/\lambda$ is the size parameter of a sphere with radius $a$ at wavelength $\lambda$.

The single scattering albedo is $a$, defined as

$$
a = \frac{Q_\mathrm{sca}}{Q_\mathrm{ext}} = \frac{Q_\mathrm{sca}}{Q_\mathrm{sca}+Q_\mathrm{abs}}
$$

and $Q_\mathrm{sca}$, $Q_\mathrm{ext}$, and $Q_\mathrm{abs}$ are the scattering, extinction, and absorption efficiencies.

Using miepython
---------------

1. You can install locally using pip::
    
    pip install --user miepython

2. or `run this code in the cloud using Google Collaboratory <https://colab.research.google.com/github/scottprahl/miepython/blob/master>`_ by selecting the Jupyter notebook that interests you.

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

Here are a few short scripts in the github repository.

* `Extinction Efficiency of Absorbing and Non-Absorbing Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/01_dielectric.py>`_ 
* `Four Micron Glass Spheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/02_glass.py>`_ 
* `One Micron Water Droplets <https://github.com/scottprahl/miepython/blob/master/miepython/examples/03_droplets.py>`_ 
* `Gold Nanospheres <https://github.com/scottprahl/miepython/blob/master/miepython/examples/04_gold.py>`_ 

Detailed documentation is available at <https://miepython.readthedocs.io>


License
-------

``miepython`` is licensed under the terms of the MIT license.
