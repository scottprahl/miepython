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

Accurate and validated Mie scattering calculations in pure Python
-----------------------------------------------------------------

``miepython`` provides a rigorously validated and efficient implementation of Mie scattering for spherical particles.  
It reproduces established reference results (including Wiscombe's MIEV0) and is designed for scientific, educational, and computational research applications in optics.

The library implements the full Mie solution, including:

- extinction, scattering, and absorption efficiencies
- asymmetry parameter (scattering anisotropy)
- angle-resolved scattering intensities
- Mie expansion coefficients
- complex amplitude functions and Mueller matrices

The implementation is numerically stable for a wide range of size parameters and refractive indices, including lossy materials and high-index contrasts.

Immediate Use in the Browser
----------------------------

The entire package can be used **immediately** in a browser — without installation — using the JupyterLite interface:

|lite|

This environment runs entirely client-side (Pyodide), and supports:

- interactive notebooks  
- real-time plotting  
- full access to ``miepython`` functions  
- reproducible experiments (downloadable notebooks)

This makes it ideal for teaching, demonstrations, or quick exploratory calculations.

Installation
------------

Install with pip:

.. code-block:: bash

    pip install miepython

Or via conda:

.. code-block:: bash

    conda install -c conda-forge miepython

Quick Start
-----------

A typical calculation is straightforward:

.. code-block:: python

    import miepython as mie

    m = 1.5 - 0.1j     # refractive index
    d = 100            # diameter (nm)
    lambda0 = 314.15   # wavelength (nm)

    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0)

Documentation and Examples
--------------------------

The full documentation is available as:

Interactive documentation (JupyterLite):
    |lite|

Static Jupyter notebooks on ReadTheDocs:
    |docs|

Among other things the documentation discusses:

- **Mathematical formulation of Mie theory**  
  https://miepython.readthedocs.io/en/latest/01_theory.html

- **Normalization conventions and units**  
  https://miepython.readthedocs.io/en/latest/02_normalization.html

- **Numerical stability considerations**  
  https://miepython.readthedocs.io/en/latest/03_stability.html

- **Validation against MIEV0 and other reference implementations**  
  https://miepython.readthedocs.io/en/latest/04_validation.html

- **Guidelines for parameter choices and truncation order**  
  https://miepython.readthedocs.io/en/latest/05_truncation.html

- **Physical interpretation, resonances, and comparison plots**  
  https://miepython.readthedocs.io/en/latest/06_examples.html

Representative results simple examples:

    https://github.com/scottprahl/miepython/tree/main/miepython/examples

.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/01.svg
.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/02.svg
.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/03.svg
.. image:: https://raw.githubusercontent.com/scottprahl/miepython/main/docs/images/04.svg

Performance and Acceleration
----------------------------

Although implemented entirely in Python, ``miepython`` supports optional Numba JIT compilation:

.. code-block:: python

    import os
    os.environ["MIEPYTHON_USE_JIT"] = "1"  # must be set before import
    import miepython

This can provide 10–50× speedups for large parameter sweeps or ensemble calculations.

Benchmark example (100,000 particles):

============ ============ ==========
Version      Time         Speedup
============ ============ ==========
Pure Python  4.00 s       1×
JIT Enabled  0.15 s       27×
============ ============ ==========

Citation
--------

If ``miepython`` contributes to your research, please cite the Zenodo archive:

DOI: 10.5281/zenodo.7949263

BibTeX:

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

``miepython`` is released under the MIT License.

