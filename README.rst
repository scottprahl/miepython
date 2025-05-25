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

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: code style: black

miepython
=========

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads| |black|

``miepython`` is a pure‑Python implementation of Mie theory for spherical
scatterers, validated against Wiscombe's reference results.  The library is
lightweight, extensively tested, and—thanks to an *optional* Numba backend—can
process nearly a million particles per second.

Overview
--------

- **Non-absorbing spheres** (dielectric particles)
- **Partially-absorbing spheres** (lossy dielectrics)  
- **Perfectly-conducting spheres** (metallic particles)

Key Features
~~~~~~~~~~~~

- ✅ **Pure Python** - No external dependencies beyond NumPy
- ✅ **Validated algorithms** - Follows Wiscombe's trusted implementation
- ✅ **Comprehensive outputs** - Extinction, scattering, backscattering, asymmetry
- ✅ **Angular distributions** - Full scattering phase functions
- ✅ **Flexible normalization** - Multiple conventions supported
- ✅ **Code Jitting** - the python Numba package enables 10-50X speedup
- ✅ **Field calculations** - Internal field coefficients coming!


Documentation
-------------

- **Full Documentation**: `miepython.readthedocs.io <https://miepython.readthedocs.io>`_
- **API Reference**: `miepython api <https://miepython.readthedocs.io/en/latest/#api-reference>`_
- **Jupyter Notebooks**: `Interactive Jupyter notebooks <https://github.com/scottprahl/miepython/tree/main/docs>`_
- **Theory Background**: `Mathematical foundations and validation <https://miepython.readthedocs.io/en/latest/07_algorithm.html>`_

Version 3.0 Breaking Changes
----------------------------

Version 3.0 introduced significant API changes and new functionality:

- **Internal field calculations** - Compute electromagnetic fields inside spheres
- **Enhanced coefficient access** - Direct access to Mie expansion coefficients
- **Future-ready architecture** - Foundation for full field calculations

If you need the old API, pin to version 2.5.5::

    pip install miepython==2.5.5

Installation
~~~~~~~~~~~~

Using pip::

    pip install miepython

Using conda::

    conda install -c conda-forge miepython

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    import miepython as mie

    # Define sphere properties
    m = 1.5 - 1j       # Complex refractive index
    d = 100            # Diameter (nm)
    lambda0 = 314.15   # Wavelength in vacuum (nm)

    # Calculate efficiencies
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


API Reference
-------------

Basic Functions
~~~~~~~~~~~~~~~

============================================ ===========================================================
Function                                     Purpose
============================================ ===========================================================
``efficiencies(m, d, lambda0, n_env=1)``     Calculate extinction, scattering, backscattering, asymmetry
``intensities(m, d, lambda0, mu, n_env=1)``  Angular scattering intensities for parallel/perpendicular polarization
``S1_S2(m, x, mu)``                          Complex scattering amplitudes
``coefficients(m, x)``                       Mie coefficients for field calculations
``phase_matrix(m, x, mu)``                   Mueller matrix for sphere
============================================ ===========================================================

Parameters
~~~~~~~~~~

- **m** (complex): Complex refractive index of sphere
- **n_env** (complex): Real refractive index of medium
- **d** (float): Sphere diameter [same units as wavelength]
- **lambda0** (float): Wavelength in vacuum [same units as diameter]
- **x** (float): Size parameter (π×diameter/wavelength)
- **mu** (array): Cosine of scattering angles


Important Conventions
---------------------

   1. **Negative imaginary refractive index**: For absorbing materials, use ``m = n - ik`` where k > 0
   2. **Albedo normalization**: Scattering phase functions integrate to the single scattering albedo over 4π steradians (customizable)

   These latter may be mitigated using custom normalization

.. code-block:: python

    # Different scattering function normalizations
    I_albedo = mie.i_unpolarized(m, x, mu, norm='albedo')  # Default
    I_unity = mie.i_unpolarized(m, x, mu, norm='one')      # Normalized to 1
    I_4pi = mie.i_unpolarized(m, x, mu, norm='4pi')        # 4π normalization



Performance & JIT Compilation
-----------------------------

``miepython`` supports **Just-In-Time (JIT) compilation** via Numba for dramatic performance improvements on large datasets. This is especially beneficial for batch calculations with thousands of particles.

Enabling JIT
~~~~~~~~~~~~

.. code-block:: python

    import os
    os.environ["MIEPYTHON_USE_JIT"] = "1"  # Must be set before importing
    import miepython as mie

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

JIT compilation provides substantial speedups for large-scale calculations:

=========== ============== ================== ==========
Version     JIT Status     Time (N=100,000)   Speedup
=========== ============== ================== ==========
v3.0.1      Disabled       4.00 seconds       1×
v3.0.1      **Enabled**    **0.15 seconds**   **27×**
=========== ============== ================== ==========

Benchmark Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import numpy as np
    from time import time

    os.environ["MIEPYTHON_USE_JIT"] = "1"  # must be before import miepython
    import miepython as mie

    # Generate random particle ensemble
    N = 100_000
    refr = np.random.uniform(1.0, 2.0, N)
    refi = np.exp(np.random.uniform(np.log(1e-4), np.log(1.0), N))
    x = np.exp(np.random.uniform(np.log(0.01), np.log(100), N))
    m = refr - 1j * refi

    # Benchmark calculation
    t0 = time()
    qext, qsca, qback, g = mie.efficiencies_mx(m, x)
    elapsed = time() - t0

    print(f"JIT enabled: {os.environ.get('MIEPYTHON_USE_JIT') == '1'}")
    print(f"Calculated {N:,} particles in {elapsed:.3f} seconds")
    print(f"Rate: {N/elapsed:,.0f} particles/second")

.. note::
   The first JIT-compiled call includes compilation overhead (~1-2 seconds). Subsequent calls achieve full performance.

Examples Gallery
----------------

The repository includes several `example scripts <https://github.com/scottprahl/miepython/tree/master/miepython/examples>`_ demonstrating different applications:

Dielectric vs. Absorbing Spheres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/01.svg
   :alt: Dielectric vs Absorbing

Glass Microspheres with Resonances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/02.svg
   :alt: Glass Spheres

Water Droplets
~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/03.svg
   :alt: Water Droplets

Gold Nanoparticles
~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/04.svg
   :alt: Gold Nanoparticles




Citing `miepython`
--------------------

If this library contributes to your research, please cite the archived release
on `zenodo <https://zenodo.org>`_

```
S. Prahl, *miepython — Pure‑Python Mie scattering calculations*, Zenodo,
16 March 2025. doi:10.5281/zenodo.7949263
```

* **Generic DOI (always the newest release)** — `10.5281/zenodo.7949263`.  The
  badge at the top of this file resolves to that record.
* **Version‑specific DOIs** — click the Zenodo badge |doi| and choose the DOI that
  corresponds to the exact version you want to cite (e.g.
  `10.5281/zenodo.14257432 for v2.5.5`).

.. code-block:: bibtex

```
@software{prahl_miepython_2025,
  author  = {Prahl, Scott},
  title   = {{miepython}: A Python library for Mie scattering calculations},
  url     = {https://github.com/scottprahl/miepython},
  doi     = {10.5281/zenodo.7949263},
  year    = {2025},
  version = {latest}
}
```

License
-------

``miepython`` is licensed under the `MIT License <LICENSE.txt>`_.

--------

**Maintained by** `Scott Prahl <https://github.com/scottprahl>`_