Changelog
=========

unreleased
-------------------
*   fix off-by-one in ``n_pole`` for ``S1_S2`` and everything built on it
    (``i_par``, ``i_per``, ``i_unpolarized``, ``phase_matrix``, ``intensities``).
    ``n_pole=1`` returned the quadrupole instead of the dipole; it now matches
    the convention already used by ``efficiencies_mx``
*   raise ``ValueError`` for an ``n_pole`` beyond the truncated series instead of
    returning zeros or an ``IndexError``
*   add multipole regression tests (series sum, closed form, per-multipole
    optical theorem) and re-execute ``docs/12_multipoles.ipynb``
*   implement the ``e_field`` argument of ``efficiencies_mx``, which was accepted
    and documented but silently ignored. With ``n_pole > 0`` it now selects the
    electric multipole a_n (``e_field=True``, the default) or the magnetic
    multipole b_n (``e_field=False``). **Behavior change:** ``n_pole > 0``
    previously returned a_n and b_n combined; that total is now
    ``e_field=True`` plus ``e_field=False``
*   make the two backends agree on ``g`` when ``n_pole > 0``. The no-JIT backend
    returned ``None`` (which became ``nan`` for array input) while the JIT
    backend returned ``0``. Both now return ``0.0``, which is the exact value:
    an isolated multipole of one parity scatters symmetrically about 90 degrees
*   fix an operator-precedence bug in the perfectly-conducting guard of ``cn_dn``.
    ``a and b or c`` made the test true for every finite index, so ``cn_dn(0j, x)``
    raised ``ZeroDivisionError`` and an infinite index produced ``nan``. The
    internal coefficients are now zero for a perfect conductor, as intended
*   drop ``fastmath`` from the numba ``_cn_dn_nb`` kernel. It implies LLVM's
    ``ninf``, which folded the ``np.isinf`` guard to ``False`` and defeated the
    fix above in the JIT backend. This also improves JIT/no-JIT agreement
*   make the internal-field coefficients ``c_n`` and ``d_n`` numerically stable.
    ``psi_n`` came from the three-term upwards recurrence, which is contaminated
    by the growing ``chi_n`` solution once n exceeds ``|mx|``.  That is the normal
    case for a relative index below one, where the coefficients were wrong by up
    to 140%.  ``psi_n`` now uses Miller's downwards recurrence seeded above ``|z|``
    and normalised against whichever of ``psi_0`` or ``psi_1`` is larger, and
    the logarithmic derivatives take the always-stable downwards route.
    Worst error against a direct SciPy evaluation over 7320 cases drops from
    90 to 3e-9, and the JIT and no-JIT backends now agree to 5e-15 where they
    previously differed by up to a factor of 39
*   always use the downwards recurrence for the logarithmic derivative ``D_n``.
    Wiscombe's criterion chose between the two recurrences from the refractive
    index alone, never from the number of terms, so it picked the upwards
    recurrence for small spheres where it is unstable.  For a lossless sphere
    ``Re(a_n)`` must equal ``|a_n|**2``; at m=1.05, x=0.1 the real parts of the
    quadrupole and higher coefficients came out with the wrong sign and up to 12
    orders of magnitude too large, which made ``efficiencies_mx(n_pole=3)``
    report a negative extinction efficiency.  That identity now holds to 4e-24
    over all lossless cases tested, and ``qext`` and ``qsca`` agree to 5e-16
    where they used to differ by 2e-7.  The pure-python backend is about 20%
    slower for a large sweep of size parameters; the numba backend is unchanged
*   remove ``fastmath`` from every numba kernel.  Besides defeating the
    perfectly-conducting guard above, its reassociation degraded the exact
    lossless identity by roughly six times relative to the pure-python backend.
    Costs about 10% on a large sweep of size parameters
*   ``an_bn`` no longer pads a zero onto the end of ``a`` and ``b``, and ``cn_dn``
    no longer computes one order more than ``an_bn``.  Both now return exactly
    Wiscombe's number of terms with every entry a real coefficient, so the
    internal and external series share one truncation.  The retained ``a_n`` and
    ``b_n`` are bit-for-bit unchanged, and the arrays are one element shorter
*   ``an_bn(m, x)`` works on both backends.  ``_an_bn_py`` defaulted ``n_pole`` to
    zero but the numba kernel's eager signature could not, so the two-argument
    call raised ``TypeError`` with the JIT on.  ``_an_bn_nb`` is now compiled
    lazily, which allows the default
*   group the arithmetic in the numba ``_an_bn_nb`` and ``_cn_dn_nb`` kernels the
    way the pure-python ones do.  Multiplying by a precomputed ``1/x`` in one and
    dividing by ``x`` in the other differed by one ulp, and the psi recurrence
    amplified that to 2.6e-7 between the backends on significant coefficients.
    They now agree to 1e-13
*   ``an_bn`` builds ``psi_n`` with Miller's downwards recurrence, the same
    treatment ``cn_dn`` already had, and builds ``xi_n`` as ``psi_n + i*chi_n``
    from an upwards ``chi_n`` recurrence.  The upwards ``psi_n`` recurrence was
    accurate enough inside Wiscombe's truncation but lost all relative accuracy
    beyond it -- about 1e-6 at the last kept order, and 1e10 or worse a few orders
    later -- so asking for extra terms returned noise.  Requesting more orders now
    converges instead: the near-field boundary mismatch at W+8 terms improves from
    4.7e-2 to 3.1e-6, and it keeps improving to W+16 rather than diverging.
    Building ``xi_n`` from the same ``psi_n`` preserves the lossless identity
    ``Re(a_n) == |a_n|**2`` at 5.6e-16.  Within the standard truncation nothing
    moves by more than 1e-11, so this is headroom rather than a correction.  The
    pure-python backend costs about 7% more for a large sweep; numba is unchanged.
    ``xi_n`` itself was already fine: it is dominated by the growing ``chi_n``
    solution, for which the upwards recurrence is the stable direction, and it
    matches SciPy to 1e-15 even twenty orders past the truncation
*   assert the scattnlay comparison instead of only plotting it.  The reference
    arrays under ``docs/data`` were used solely by ``docs/15_2D_fields.ipynb``,
    which printed relative errors without checking them, so a near-field
    regression was invisible; the median error improved 19x during this release
    and nothing would have noticed either that or its reversal.
    ``tests/test_scattnlay_reference.py`` now checks the scalar efficiencies,
    which are inlined like the existing MIEV0 values and need no data files, and
    compares both 121x121 field grids point by point.  Grid points that land
    exactly on the sphere surface are excluded, since the radial E component is
    discontinuous there and the two codes may put them on opposite sides
*   ``docs/15_2D_fields.ipynb`` reads the reference arrays from ``docs/data``
    rather than downloading them from the published branch, so ``make note-test``
    works offline and checks the working tree instead of the last release.  Its
    stored error figures were also stale and have been refreshed
*   collapse the per-backend test files.  ``test_jit.py``/``test_nojit.py`` and
    ``test_jit_abcd.py``/``test_nojit_abcd.py`` became ``test_mie.py`` and
    ``test_abcd.py`` for the high-level API, which needs one copy because
    ``core.py`` has no backend-specific code, plus ``test_kernels.py`` for the
    kernel-level tests, which the ``kernels`` fixture runs against both backends
    in one process.  All 74 distinct test names are preserved, and the merge picks
    up what had drifted between the two halves: an extra MIEV0 conducting case,
    four extra ``qext`` assertions, and both spellings of a perfectly conducting
    index.  Shared reference implementations moved to ``tests/reference.py``
*   drop the ``MIEPYTHON_RUN_MIE_SPEED`` switch and ``test_mie_backend_speed.py``.
    Nothing ever set that variable, so the three real assertions in that file --
    a backend agreement check over 4000 random particles and 361 angles -- had
    never run, while the four assertions it also gated were tautologies
    (``elapsed > 0``).  The agreement check moved to ``test_backend_parity.py``
    where it always runs and costs about a tenth of a second, and the speedup
    report it printed became ``benchmark_efficiencies.py --compare``, which
    ``make speed`` now calls
*   the benchmark scripts ``test_jit_speed.py`` and ``test_nojit_speed.py`` became
    one ``tests/benchmark_efficiencies.py``.  They timed at import rather than in a
    test function, so pytest spent about seven seconds running benchmarks during
    collection while collecting nothing; a full test pass is now much quicker.
    They also both timed whichever backend happened to be bound, so the JIT-named
    one could report pure-python speed.  ``make speed`` runs the single script once
    per backend
*   add ``tests/test_backend_parity.py``, which compares the two kernel sets
    directly -- signatures and values -- in a single process, and merge the
    ``test_*_D.py`` pair into one ``test_D.py`` driven by a ``kernels`` fixture
    that parametrizes over both backends.  Tests written this way need only one
    copy instead of a JIT/no-JIT pair
*   test ``monte_carlo.py``, which had no tests at all, and drop the unreachable
    index clamp in ``generate_mie_costheta``: ``numpy.random.random`` draws from
    [0, 1), so the index can never run off the table.  A test forces the largest
    float below one to show that.  The sampler is checked against physics rather
    than against itself: the mean of the drawn cosines reproduces the asymmetry
    parameter g, computed by a completely separate path, and the empirical
    distribution follows the inverse-transform table.  With this the whole package
    is covered, so ``fail_under`` is now 100
*   record that ``cdf`` overshoots one.  It sums ``i_unpolarized / num`` instead of
    multiplying by the true spacing 2/(num-1), so the last value is about
    1 + 1.3/num: at the num=20 of ``docs/06_random_deviates.ipynb`` the cumulative
    distribution reaches 1.067 even though the docstring promises a maximum of 1.
    A test pins the 1/num convergence so the quadrature cannot quietly change
*   test the kernel paths an ordinary Mie call never reaches, taking both
    ``mie_jit.py`` and ``mie_nojit.py`` to 100%.  ``_D_upwards`` is kept for
    comparison but ``D_calc`` no longer selects it, so it is now checked against
    SciPy where it is valid and against ``_D_downwards``.  Also covered: the
    overflow rescale inside ``_psi_downwards``, which a small argument triggers and
    which the normalisation afterwards has to undo exactly; ``cn_dn`` for a sphere
    of no size; and the ``m=0`` shorthand for a perfect conductor, now shown to be
    identical to passing ``1-10000j``.  Every module except ``monte_carlo.py`` is
    fully covered, with no partial branches left anywhere
*   test the scattered-only fields and the explicit term count in ``field.py``,
    taking it to 100%.  Nothing had ever called a near-field routine with
    ``include_incident=False`` for H, or passed ``n_pole`` at all.  Outside the
    sphere the total field minus the scattered field is now checked against an
    incident plane wave written out independently of miepython, and inside the
    sphere ``include_incident`` is confirmed to make no difference
*   test the array dispatch and error paths in ``core.py``, taking it to 100%.
    The ``RuntimeError`` for mismatched ``m`` and ``x`` lengths, the one for array
    input without ``n_pole``, the ``internal`` variants of ``coefficients`` and the
    mixed scalar-and-array calls were all untested, as was the rejection of an
    unknown normalization.  Every array result is now checked against the
    equivalent scalar calls
*   fix ``S1_S2`` rejecting a list or an integer array of angles under numba.  The
    kernel is declared for ``float64[:]``, so ``S1_S2(m, x, [0, 1])`` worked with the
    JIT off and raised ``TypeError`` with it on; ``core.py`` now coerces the angles
    once, at the boundary, so both backends take the same arguments
*   test the missing-SciPy path in ``__init__.py`` and the remaining parts of
    ``rayleigh.py``, taking them from 52% and 72% to 100%; the total reaches 95%.
    The SciPy fallback is what keeps the package usable in JupyterLite and had never
    been exercised: the tests re-import the package behind an import hook, check the
    placeholders name themselves and chain the original error, and confirm that a
    failure other than SciPy still propagates.  ``rayleigh``'s physical-units
    wrappers, all thirteen normalization spellings and its error paths were also
    untested
*   record that ``rayleigh``'s normalization closes only to O(x**2): ``qsca`` in
    ``efficiencies_mx`` stops at x**4 while the ``a1`` used by ``S1_S2`` carries an
    x**5 term, so the 'one' normalization integrates to 1 + 0.07 x**2 rather than 1.
    A test pins that order so extending one expansion without the other shows up
*   test ``vsh.py`` and ``util.py``, which the new coverage run showed at 60% and
    55%.  Both are now at 100% branch coverage, and the total rises from 86% to
    91%.  The four ``M_*_array``/``N_*_array`` helpers, the ``deg=True`` angle
    paths and the small-argument series in ``N_base`` had no tests at all.  One of
    the new tests checks ``vsh.py`` against ``field.py``, which holds a second copy
    of the same vector spherical harmonics, so the two can no longer drift apart
*   measure coverage.  ``make coverage`` runs the suite once per backend and
    combines the results, and a CI job publishes the HTML report as an artifact.
    Naive coverage is misleading here: coverage.py cannot see inside numba's
    compiled functions, so ``mie_jit.py`` reports 11% no matter how well it is
    tested.  The second pass therefore sets ``NUMBA_DISABLE_JIT=1``, which runs the
    njit bodies as plain Python and lifts that file to 96%.  The combined figure is
    86% with branch coverage, and ``fail_under`` is set to 84 as a floor to ratchet
    upward.  Two tests that deliberately assert numba is compiled were rewritten to
    hold in either mode rather than being skipped, so all three modes stay green
*   widen the CI matrix.  It covered ubuntu on Python 3.10 and 3.14 only, and
    because it also ran just two test files, most of the suite had never executed
    on the oldest supported Python.  Linux now runs every version the classifiers
    promise, 3.10 through 3.14, and macOS and Windows run both ends of that range,
    since numba holds the platform-specific half of the package.  numba 0.66 ships
    wheels for all of those combinations
*   discover example scripts and notebooks relative to the test file rather than
    the working directory.  ``test_all_examples.py`` globbed a relative path, so a
    run started anywhere but the repository root collected nothing and reported
    success; it now asserts that it found something
*   CI now lints.  It only ran pytest, so a pull request with formatting or lint
    errors passed clean and the checks in ``make rcheck`` were enforced on nobody.
    A new job runs ``make lint``, the same target a release check uses, through uv,
    so the list of checks lives in the Makefile alone and cannot drift from CI
*   add a ``make lint`` target covering ruff, black, pylint, rstcheck, yamllint,
    check-manifest and pyroma, and have ``make rcheck`` delegate to it.  That also
    closes a gap: ``yaml-check`` existed but was never part of ``rcheck``, so the
    workflow files were only ever checked by hand
*   the ``test_jit*`` files now really exercise the numba backend.  ``_backend.py``
    binds its kernels the first time ``miepython`` is imported, so setting
    ``MIEPYTHON_USE_JIT`` inside a test module only worked when that module
    happened to import the package first.  In a full ``make test`` run something
    always imported it earlier with the JIT off, so those files silently retested
    the pure-python kernels and passed.  ``tests/conftest.py`` now reads the
    backend that actually got bound and skips the files belonging to the other
    one, and ``make test`` runs the suite once per backend.  The backend-agnostic
    tests run under both, which is new coverage for them.  New ``make test-jit``
    and ``make test-nojit`` targets run a single pass, and CI now runs the whole
    suite per backend rather than one file each
*   raise a clear error instead of returning ``inf`` or ``nan`` when a scattering
    function is asked to normalise against a sphere that does not scatter.  A
    sphere whose index matches its surroundings gave ``i_unpolarized`` values of
    ``inf`` for the 'albedo', 'one' and '4pi' normalizations and a bare
    ``ZeroDivisionError`` for 'qext'.  The 'wiscombe', 'bohren' and 'qsca'
    choices do not divide by an efficiency and still work
*   ``efficiencies_mx`` returns zero rather than ``nan`` for ``qback`` when the
    size parameter is zero.  The small-sphere form of ``qback`` is 0/0 there,
    although its limit is zero, and a single zero in an array of size parameters
    used to leave a ``nan`` behind
*   fix ``util.cartesian_to_spherical``, which raised on array input because of a
    scalar ``if r != 0`` test.  It now broadcasts its three arguments, reports
    theta as zero at the origin, and clips ``z/r`` so rounding cannot push the
    arccos argument out of range.  ``field.py`` dropped its private duplicate of
    this routine
*   fix ``util`` exporting ``_all_`` instead of ``__all__``, and list
    ``phasor_str_scalar`` alongside the rest
*   the near-field routines keep two more terms than Wiscombe's criterion.  That
    criterion truncates the scattered series, which converges faster than the
    field evaluated right at the sphere surface: the tangential E and H mismatch
    across the boundary drops from 1.5e-5 to 3.2e-6, and the median disagreement
    with scattnlay over a 121x121 slice falls from 1.1e-11 to 3.3e-14.  Further
    orders gain almost nothing once ``psi_n`` is computed stably, so two is where
    it stops.  The scattering quantities are untouched, and ``e_far`` still uses
    the criterion unchanged.  ``miepython.core.wiscombe_terms`` names the count
*   ``pi_tau`` now fills ``tau`` for the highest order.  It had always left the
    last entry zero, which was hidden by the ``an_bn`` padding above and would
    have silently dropped a term once the padding went away.  ``n_pole`` may now
    address the highest order, which the old bounds check rejected

3.2.0 (03/06/2026)
-------------------
*   fix error in E & H calculations in the near field when y≠0 (thanks @dorianherle)
*   add regression test

3.1.0 (02/07/2026)
-------------------
*   add near-field E and H field APIs, fix boundary continuity/medium handling, and expand validation tests
*   add field calculation utilities, field module cleanup/help text improvements, and precomputed E/H reference data (via scattnlay)
*   improve performance: ~20% speedups in Mie backends and faster near-field calculations; add speed benchmarks
*   documentation updates: new/updated notebooks (boundary conditions, 2D fields, performance), clarified conventions
*   refresh docs/README visuals and assets; add custom CSS for docs images
*   jupyterlite/RTD updates: config moves, build modernizations, and avoid numba install in JupyterLite
*   packaging/CI: pyproject and requirements cleanup, improved PyPI workflow, updated citation automation/config
*   misc cleanups: remove unused files/images, minor typos, Makefile and docs config tidy-ups, CITATION.cff refresh

3.0.5 (1/2/2026)
-------------------
*   fix versioning

3.0.4 (1/2/2026)
-------------------
*   Host jupyterlite instance of github
*   improve citation guidelines
*   add test for unpolarized intensity methods
*   begin work on local electrical and magnetic fields
*   using black now with longer lines
*   improved packaging
*   improve readme

3.0.3
-------------------
*   skipped

3.0.2 (5/25/2025)
-------------------
*   fix version number

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
