.. |pypi| image:: https://img.shields.io/pypi/v/miepython?color=68CA66
   :target: https://pypi.org/project/miepython/
   :alt: PyPI

.. |github| image:: https://img.shields.io/github/v/tag/scottprahl/miepython?label=github&color=68CA66
   :target: https://github.com/scottprahl/miepython
   :alt: GitHub

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/miepython?label=conda&color=68CA66
   :target: https://github.com/conda-forge/miepython-feedstock
   :alt: Conda

.. |doi| image:: https://zenodo.org/badge/99259684.svg
   :target: https://zenodo.org/badge/latestdoi/99259684
   :alt: DOI

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

.. |lite| image:: https://img.shields.io/badge/try-JupyterLite-68CA66.svg
   :target: https://scottprahl.github.io/miepython/
   :alt: Try JupyterLite

miepython
=========

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads|

|lite|

**Fast, pure-Python Mie scattering calculations for spherical particles**

``miepython`` implements Mie theory for light scattering by spherical particles. It's validated against Wiscombe's reference implementation and can process nearly a million particles per second with optional Numba acceleration.

Perfect for modeling aerosols, colloids, nanoparticles, and atmospheric optics.

Features
--------

✅ **Pure Python** — Works anywhere, no compilers needed  
✅ **Fast** — Optional Numba JIT gives 10-50× speedup  
✅ **Validated** — Matches Wiscombe's MIEV0 reference code  
✅ **Complete** — Efficiencies, angular patterns, phase matrices  
✅ **Flexible** — Multiple normalization conventions  
✅ **Well-documented** — Extensive examples and theory guides

Supported Particles
~~~~~~~~~~~~~~~~~~~

- Non-absorbing spheres (dielectrics)
- Partially-absorbing spheres (lossy materials)  
- Perfectly-conducting spheres (metals)

Installation
------------

**Using pip:**

.. code-block:: bash

    pip install miepython

**Using conda:**

.. code-block:: bash

    conda install -c conda-forge miepython

**Immediately using jupyter-lite.**

  No installation needed!  |lite| **

Quick Start
-----------

.. code-block:: python

    import miepython as mie

    # 100 nm sphere in air at 314.15 nm wavelength
    m = 1.5 - 0.1j     # Complex refractive index (n - ik)
    d = 100            # Diameter (nm)
    lambda0 = 314.15   # Wavelength (nm)

    # Calculate scattering properties
    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0)

    print(f"Extinction efficiency:  {qext:.3f}")
    print(f"Scattering efficiency:  {qsca:.3f}")  
    print(f"Backscatter efficiency: {qback:.3f}")
    print(f"Scattering anisotropy:  {g:.3f}")

**Output:**

.. code-block:: text

    Extinction efficiency:  2.336
    Scattering efficiency:  0.663
    Backscatter efficiency: 0.573
    Scattering anisotropy:  0.192

Performance
-----------

Enable Numba JIT compilation for dramatic speedups on large datasets:

.. code-block:: python

    import os
    os.environ["MIEPYTHON_USE_JIT"] = "1"  # Set before importing
    import miepython as mie

**Benchmark Results** (100,000 particles):

============ ============ ==========
Version      Time         Speedup
============ ============ ==========
Pure Python  4.00 s       1×
**With JIT** **0.15 s**   **27×**
============ ============ ==========

Core API
--------

**Efficiency Calculations**

.. code-block:: python

    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0, n_env=1)

**Angular Scattering**

.. code-block:: python

    # Scattering intensities vs angle
    mu = np.cos(np.linspace(0, np.pi, 100))  # Scattering angles
    Ipar, Iper = mie.intensities(m, d, lambda0, mu)

**Advanced Functions**

============================================ ===========================================================
Function                                     Purpose
============================================ ===========================================================
``efficiencies(m, d, lambda0, n_env=1)``     Extinction, scattering, backscattering, asymmetry
``intensities(m, d, lambda0, mu, n_env=1)``  Angular scattering for parallel/perpendicular polarization
``S1_S2(m, x, mu)``                          Complex scattering amplitudes
``coefficients(m, x)``                       Mie expansion coefficients
``phase_matrix(m, x, mu)``                   Full Mueller scattering matrix
============================================ ===========================================================

**Parameters:**

- **m** (complex): Refractive index of sphere (use ``n - ik`` with k > 0 for absorption)
- **d** (float): Sphere diameter  
- **lambda0** (float): Wavelength in vacuum (same units as diameter)
- **n_env** (float): Refractive index of surrounding medium (default: 1.0)
- **mu** (array): Cosine of scattering angles
- **x** (float): Size parameter = πd/λ

Examples
--------

The repository includes detailed `example notebooks <https://github.com/scottprahl/miepython/tree/master/docs>`_ showing:

**Dielectric vs. Absorbing Spheres**

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/01.svg
   :alt: Dielectric vs Absorbing

**Glass Microspheres (showing resonances)**

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/02.svg
   :alt: Glass Spheres

**Water Droplets**

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/03.svg
   :alt: Water Droplets

**Gold Nanoparticles (plasmonic resonance)**

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/04.svg
   :alt: Gold Nanoparticles

Documentation
-------------

📚 **Full Documentation:** `miepython.readthedocs.io <https://miepython.readthedocs.io>`_

📖 **Interactive Tutorials:** `Jupyter notebooks <https://github.com/scottprahl/miepython/tree/main/docs>`_

🔬 **Theory & Validation:** `Mathematical foundations <https://miepython.readthedocs.io/en/latest/07_algorithm.html>`_

Version 3.0 Changes
-------------------

Version 3.0 rationalized the API and added new capabilities:

- Direct access to Mie expansion coefficients
- Improved function naming consistency  
- Foundation for future field calculations

If you need the old API, pin to version 2.5.5:

.. code-block:: bash

    pip install miepython==2.5.5

Citation
--------

If ``miepython`` helps your research, please cite it via Zenodo:

**Generic DOI** (always latest): `10.5281/zenodo.7949263 <https://doi.org/10.5281/zenodo.7949263>`_

**BibTeX:**

.. code-block:: bibtex

    @software{prahl_miepython_2025,
      author  = {Prahl, Scott},
      title   = {{miepython}: A Python library for Mie scattering calculations},
      url     = {https://github.com/scottprahl/miepython},
      doi     = {10.5281/zenodo.7949263},
      year    = {2025}
    }

License
-------

``miepython`` is licensed under the `MIT License <LICENSE.txt>`_.