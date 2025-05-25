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

**A pure Python library for Mie light scattering calculations**

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads| |black|

--------

Overview
--------

``miepython`` is a pure‑Python implementation of Mie theory for spherical
scatterers, validated against Wiscombe's reference results.  The library is
lightweight, extensively tested, and—thanks to an *optional* Numba backend—can
process nearly a **million particles per second**.

- **Non-absorbing spheres** (dielectric particles)
- **Partially-absorbing spheres** (lossy dielectrics)  
- **Perfectly-conducting spheres** (metallic particles)

Full documentation at <https://miepython.readthedocs.io>

Key Features
~~~~~~~~~~~~

- ✅ **Pure Python** - No external dependencies beyond NumPy
- ✅ **Validated algorithms** - Follows Wiscombe's trusted implementation
- ✅ **Comprehensive outputs** - Extinction, scattering, backscattering, asymmetry
- ✅ **Angular distributions** - Full scattering phase functions
- ✅ **Field calculations** - Internal field coefficients (v3.0+)
- ✅ **Flexible normalization** - Multiple conventions supported
- ✅ **Code Jitting** - the python Numba package enables 10-50X speedup

Quick Start
-----------

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

Angular Scattering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate scattering at different angles
    angles = np.linspace(0, np.pi, 100)
    mu = np.cos(angles)

    # Get parallel and perpendicular intensities
    I_par, I_per = mie.intensities(m, d, lambda0, mu)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.semilogy(np.degrees(angles), I_par, label='Parallel')
    plt.semilogy(np.degrees(angles), I_per, label='Perpendicular')
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Intensity (1/sr)')
    plt.legend()
    plt.show()

API Reference
-------------

Core Functions
~~~~~~~~~~~~~~

=============================================== ===========================================================
Function                                        Purpose
=============================================== ===========================================================
``efficiencies(m, d, lambda0)``                 Calculate extinction, scattering, backscattering, asymmetry
``intensities(m, d, lambda0, mu)``              Angular scattering intensities for parallel/perpendicular polarization
``S1_S2(m, x, mu)``                             Complex scattering amplitudes
``coefficients(m, x)``                          Mie coefficients for field calculations
=============================================== ===========================================================

Parameters
~~~~~~~~~~

- **m** (complex): Refractive index of sphere relative to medium
- **d** (float): Sphere diameter [same units as wavelength]
- **lambda0** (float): Wavelength in vacuum [same units as diameter]
- **x** (float): Size parameter (π×diameter/wavelength)
- **mu** (array): Cosine of scattering angles

Size Parameter Functions
~~~~~~~~~~~~~~~~~~~~~~~~

For direct size parameter calculations:

.. code-block:: python

    x = np.pi * diameter / wavelength
    qext, qsca, qback, g = mie.efficiencies_mx(m, x)

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

Important Conventions
---------------------

.. warning::
   **Key assumptions in miepython:**

   1. **Negative imaginary refractive index**: For absorbing materials, use ``m = n - ik`` where k > 0
   2. **Albedo normalization**: Scattering phase functions integrate to the single scattering albedo over 4π steradians (customizable)

   These conventions may differ from other Mie codes - always verify when comparing results!

Version 3.0 Breaking Changes
----------------------------

Version 3.0 introduces significant API changes and new functionality:

New Features
~~~~~~~~~~~~

- **Internal field calculations** - Compute electromagnetic fields inside spheres
- **Enhanced coefficient access** - Direct access to Mie expansion coefficients
- **Future-ready architecture** - Foundation for full field calculations

Migration
~~~~~~~~~

If you need the old API, pin to version 2.5.5::

    pip install miepython==2.5.5

For new projects, use v3.0+ to access the latest features and improvements.

Advanced Usage
--------------

Multiple Wavelengths
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Wavelength-dependent calculations
    wavelengths = np.linspace(400, 700, 100)  # nm
    diameters = np.full_like(wavelengths, 200)  # 200 nm spheres

    results = []
    for wl, d in zip(wavelengths, diameters):
        qext, qsca, qback, g = mie.efficiencies(m, d, wl)
        results.append([qext, qsca, qback, g])

    results = np.array(results)

Custom Normalization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Different scattering function normalizations
    I_albedo = mie.i_unpolarized(m, x, mu, norm='albedo')  # Default
    I_unity = mie.i_unpolarized(m, x, mu, norm='one')      # Normalized to 1
    I_4pi = mie.i_unpolarized(m, x, mu, norm='4pi')        # 4π normalization

Documentation
-------------

- **Full Documentation**: `miepython.readthedocs.io <https://miepython.readthedocs.io>`_
- **API Reference**: Complete function documentation with examples
- **Theory Background**: Mathematical foundations and validation
- **Example Gallery**: Interactive Jupyter notebooks

Citation
--------

If you use miepython in your research, please cite:

.. code-block:: bibtex

    @software{miepython,
      author = {Scott Prahl},
      title = {miepython: A Python library for Mie scattering calculations},
      url = {https://github.com/scottprahl/miepython},
      doi = {10.5281/zenodo.xxxxx},
      year = {2024}
    }

Contributing
------------

Contributions are welcome! Please see our `contributing guidelines <CONTRIBUTING.md>`_ for details on:

- Reporting bugs
- Suggesting enhancements  
- Submitting pull requests
- Running tests locally

Support
-------

- **Issues**: `GitHub Issues <https://github.com/scottprahl/miepython/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/scottprahl/miepython/discussions>`_
- **Documentation**: `ReadTheDocs <https://miepython.readthedocs.io>`_

License
-------

``miepython`` is licensed under the `MIT License <LICENSE.txt>`_.

--------

**Maintained by** `Scott Prahl <https://github.com/scottprahl>`_