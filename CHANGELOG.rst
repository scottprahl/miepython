Changelog
=========

3.0.1 (5/25/2025)
-------------------
*   fix JIT regression (thanks @avgeiss)
*   clarify polarization in docstrings
*   improve README.rst
*   fix git branches
*   rename mie.mie_scalar to mie.single_sphere
*   rename small_mie_sphere to small_sphere
*   rename small_conducting_mie to small_conducting_sphere
*   rationalize importing of jit and non-jit code
*   add test_jit_speed.py and test_nojit_speed.py

3.0.0 (3/16/2025)
-------------------
*   breaking api changes
*   api is more sane: mie.efficiencies() instead of miepython.ez_mie()
*   use core.py to cleanly separate jit and non-jit code
*   new function to calculate mie coefficients inside sphere
*   new function to calculate E-fields near and far from sphere (only works in far-field)
*   new rayleigh.py
*   new vsh.py to calculate vector spherical harmonics
*   new util.py for printing complex numbers
*   new bessel.py for complete spherical bessel function support
*   put Monte Carlo routines into their own file
*   use black for python formatting
*   update all notebooks to use new api
*   add more tests


2.5.5 (12/01/2025)
-------------------
*   add support for specific spherical modes
*   only branch is now 'main'

2.5.4 (05/07/2024)
-------------------
*   add auto-dating in CITATION
*   add python 3.12
*   add ruff, pylint config
*   simplify mie_cdf and fix notebook
*   set 3.7 as earliest python version

v2.5.3 (8/5/2023)
-------------------
*   conda-forge fails because test files are not included

v2.5.1 (8/5/2023)
-------------------
*   change tests to accommodate conda-forge
*   require python>=3.9 to accommodate latest numba
*   get rid of tox

v2.5.0 (8/4/2023)
-------------------
*   fix scattering function for very small spheres

v2.4.0 (6/10/2023)
-------------------
*   add mie_phase_matrix() to calculate scattering (Mueller) matrix

v2.3.2
-------------------
*   fix typo in README.rst that prevented pypi upload
*   add CITATION.cff to base level of miepython repository

v2.3.1
-------------------
*   add DOI for citation purposes

v2.3.0
-------------------
*   add optional argument to change scattering function normalization
*   document normalization in new notebook
*   store data in correct place
*   store version __init__.py so scripts can query it
*   fix typo in header of gold sphere example script
*   remove workaround for older Sphinx version

v2.2.3 (1/26/2022)
------------------
*   update _mie_An_Bn in miepython_nojit
*   store data in module so github testing passes
*   fix build of API documentation on miepython.readthedocs.io
*   only test back to python 3.9 because importlib.resources
*   3.9 is only needed for a few of the jupyter notebooks

v2.2.2 (1/25/2022)
------------------
*   modify _mie_An_Bn to allocate and return An and Bn
*   fix minor packaging issue
*   explicitly define encoding when opening files
*   explicitly use .readthedocs.yaml to build docs
*   use rtd theme for docs
*   add docs/requirements
*   restrict Jinja2 to 2.11.3 in docs/requirements

v2.2.1 (9/5/2021)
-----------------
*   create pure python packages
*   include wheel file
*   package as python3 only

2.1.0 (05/22/21)
----------------
*   fix case when scalar angle used with mie_S1_S2()
*   add pypi badge
*   fix notebook testing
*   thanks to @zmoon for the following changes:
*   add requirements-dev.txt
*   add example script testing
*   add workflow testing
*   fix Au/Ag error
*   fix examples that use refractiveindex.info
*   add testing badge

2.0.1 (04/25/21)
----------------
*   fix packaging mistake

2.0.0 (04/25/21)
----------------
*   use numba for 10-700X speed improvement
*   thanks to @jbecca and @pscicluna for their help
*   add performance notebook
*   add automated notebook checking
*   test more code

1.3.3 (03/21/21)
----------------
*   colab badge and link
*   change theme for sphinx documentation
*   add requirements.txt to avoid installing sphinx
*   fix restructured text errors
*   advise everywhere to `pip install --user miepython` to avoid permission problems

1.3.2 (01/13/21)
----------------
*   add ez_mie(m, d, lambda0)
*   add ez_intensities(m, d, lambda0, mu)
*   fix formatting
*   fix api autodoc
*   specify newer pythons
*   better install instructions

1.3.1 (03/30/20)
----------------
*   improve docstrings
*   use Sphinx documentation
*   host docs on readthedocs.io
*   use tox

1.3.0 (02/19/19)
----------------
*   fix calculations for small spheres (x<0.05)
*   added notebook doc/09_backscattering.ipynb
*   general tweaks to documentation throughout
*   improved README.md

1.2.0. (02/08/19)
-----------------
*   fix bug so that large sphere calculations work correctly
*   add tests for large spheres
*   add tests for backscattering efficiency
*   add documentation notebook for large spheres
*   add direct links to documentation
*   finish fixing fractions in notebooks
*   improve README.md

1.1.1. (06/25/18)
------------------
*   fix github rendering of fractions in equations
*   add developer instructions
*   fix fractions for github
*   add missing doc files found my check-manifest
*   setup.py fixes suggested by pyroma
*   pep8 compliance and delinting using pylint
*   add missing doc files found my check-manifest
*   setup.py fixes suggested by pyroma
*   pep8 compliance and delinting using pylint
*   update version
*   add notebook doc/08_large_spheres.ipynb

1.1.0 (03/02/2018)
------------------
*   update version
*   initial commit of 04_rayleigh.ipynb
*   renamed doc files
*   use new functions from miepython
*   omit low level tests
*   add __author__ and __version__
*   add i_par, i_per, i_unpolarized, and hide private functions
*   rename doc files
*   add quantitative comparisons of angular scattering
*   tweak verbiage
*   ignore more
*   initial commit
*   more cleanup
*   ignore dist files
*   minor reorg of contents
*   fix typos, add more refraction stuff
*   Changes to match PEP8 style
*   add minor comments, fix typos

1.0.0 (08/27/2017)
------------------
*   Added docs in form of Jupyter notebooks

0.4.2 (08/26/2017)
------------------
*   messed up github release 0.4.1

0.4.1 (08/26/2017)
------------------
*   fix typo

0.4.0 (08/26/2017)
------------------
*   update README to include basic testing
*   mie(m,x) work automatically with arrays
*   adding MANIFEST.in so examples get included

0.3.2 (07/07/2017)
------------------
*   update README, bump to 0.3.2
*   Fix examples so they work.

0.3.1 (07/07/2017)
------------------
*   Bump version.
*   Add functions to __init__.py.

0.3.0 (07/07/2017)
------------------
*   Update README again.
*   Update README.
*   More packaging issues.
*   Only include normalized scattering functions.
*   Tweak setup.py and add .gitignore.
*   Rename README.
*   Add small sphere calc for S1 and S2.
*   Label tests with MIEV0 cases.
*   Rename example.
*   Add gold sphere example.
*   Add a few example programs.
*   Remove unused tests.
*   Remove extraneous ; simplify test.py, add tests.
*   Simplify test suite management.
*   Rename awkward test_miepython to just test.
*   Reorganize tests, add S1 & S2 test.
*   Added capabilities. Barely working test suite.
*   Add more tests that fail.
*   Move files around.
*   Add boilerplate files and start adding unit tests.
*   Rename to miepython.
*   Initial check in.
