.. |pypi-badge| image:: https://img.shields.io/pypi/v/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: pypi

.. |github-badge| image:: https://img.shields.io/github/v/tag/scottprahl/miepython?label=github&color=68CA66
   :target: https://github.com/scottprahl/miepython
   :alt: github

.. |conda-badge| image:: https://img.shields.io/conda/vn/conda-forge/miepython?label=conda&color=68CA66
   :target: https://github.com/conda-forge/miepython-feedstock
   :alt: conda

.. |doi-badge| image:: https://zenodo.org/badge/99259684.svg
   :target: https://zenodo.org/badge/latestdoi/99259684
   :alt: doi

.. |license-badge| image:: https://img.shields.io/github/license/scottprahl/miepython?color=68CA66
   :target: https://github.com/scottprahl/miepython/blob/master/LICENSE.txt
   :alt: License

.. |testing-badge| image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml
   :alt: Testing

.. |docs-badge| image:: https://readthedocs.org/projects/miepython/badge?color=68CA66
   :target: https://miepython.readthedocs.io
   :alt: Docs

.. |downloads-badge| image:: https://img.shields.io/pypi/dm/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: Downloads

miepython
=========

by Scott Prahl

|pypi-badge| |github-badge| |conda-badge| |doi-badge|

|license-badge| |testing-badge| |docs-badge| |downloads-badge|


``miepython`` is a pure Python module to calculate light scattering for
non-absorbing, partially-absorbing, or perfectly-conducting spheres. Mie
theory is used, following `the procedure described by Wiscombe
<http://opensky.ucar.edu/islandora/object/technotes:232>`_. This code has
been validated against his work. 

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/mie-diagram2.svg
   :width: 700px
   :alt: scattering diagram

This code provides functions for calculating the extinction efficiency,
scattering efficiency, backscattering, and scattering asymmetry.
Moreover, a set of angles can be given to calculate the scattering at various
angles for a sphere.

When comparing different Mie scattering codes, make sure that you're
aware of the conventions used by each code.  ``miepython`` makes the
following assumptions

* the imaginary part of the complex index of refraction for absorbing spheres is *negative*.

* the scattering phase function is normalized so it equals the *single scattering albedo* when integrated over 4π steradians by default.  This normalization can be changed (see the normalization notebook for details).

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

Usage for those that don't do Jupyter
--------------------------------------

Basic Mie Calculations
^^^^^^^^^^^^^^^^^^^^^^^

    from miepython import mie
    
    complex_refractive_index = 1.5-1j    # convention is negative imaginary part
    size_parameter = 1                   # 2𝜋(radius)/λ
    qext, qsca, qback, g = mie(complex_refractive_index, size_parameter)

    print("The extinction efficiency  is %.3f" % qext)
    print("The scattering efficiency  is %.3f" % qsca)
    print("The backscatter efficiency is %.3f" % qback)
    print("The scattering anisotropy  is %.3f" % g)

should produce::

    The extinction efficiency  is 2.336
    The scattering efficiency  is 0.663
    The backscatter efficiency is 0.573
    The scattering anisotropy  is 0.192


Simple Dielectric
^^^^^^^^^^^^^^^^^^

The script `01_dielectric.py <https://raw.githubusercontent.com/scottprahl/miepython/master/miepython/examples/01_dielectric.py>`_

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/01_plot.svg

Glass Spheres
^^^^^^^^^^^^^^

The script `02_glass.py <https://raw.githubusercontent.com/scottprahl/miepython/master/miepython/examples/02_glass.py>`_

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/02_plot.svg

Water Droplets
^^^^^^^^^^^^^^^

The script `03_droplets.py <https://raw.githubusercontent.com/scottprahl/miepython/master/miepython/examples/03_droplets.py>`_

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/03_plot.svg

Small Gold Spheres
^^^^^^^^^^^^^^^^^^^

The script `04_gold.py <https://raw.githubusercontent.com/scottprahl/miepython/master/miepython/examples/04_gold.py>`_

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/master/docs/04_plot.svg


Usage for those that use Jupyter
---------------------------------

All the Jupyter notebooks are available in the docs directory and they are all viewable at <https://miepython.readthedocs.io>


Script Examples for those that don't do Jupyter
-----------------------------------------------

All the Jupyter notebooks are in the docs directory and shown at <https://miepython.readthedocs.io>

You can also use a Jupyter notebook immediately (well, you do have wait a bit for everything to get uploaded) by clicking the Google Colaboratory button below

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/scottprahl/miepython/blob/master
  :alt: Colab


License
-------

``miepython`` is licensed under the terms of the MIT license.
