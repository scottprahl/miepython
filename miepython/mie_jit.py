"""
Low-level Mie calculations that use numba.
"""

import numpy as np
from numba import njit, complex128, float64, int64, boolean

__all__ = (
    "_D_calc_nb",
    "_an_bn_nb",
    "_cn_dn_nb",
    "_pi_tau_nb",
    "_S1_S2_nb",
    "_single_sphere_nb",
    "_small_conducting_sphere_nb",
    "_small_sphere_nb",
)


@njit((complex128, int64), cache=True)
def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function

    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    zinv = 2.0 / z
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = 1.0 / alpha_j1 + aj
        alpha_j2 = 1.0 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -1
        runratio = ratio * runratio

    return -N / z + runratio


@njit((complex128, int64, complex128[:]), cache=True)
def _D_downwards(z, N, D):
    """
    Compute the logarithmic derivative by downwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
    """
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        last_D = n / z - 1.0 / (last_D + n / z)
        D[n - 1] = last_D


@njit((complex128, int64, complex128[:]), cache=True)
def _D_upwards(z, N, D):
    """
    Compute the logarithmic derivative by upwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
    """
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z


@njit((complex128, int64), cache=True)
def _psi_downwards_nb(z, nstop):
    """
    Compute the Riccati-Bessel functions psi_0(z)..psi_nstop(z).

    The three-term upwards recurrence
    ``psi_{n+1} = (2n+1)/z psi_n - psi_{n-1}`` is contaminated by the growing
    chi_n solution once n exceeds |z|, which is the normal case for psi_n(mx)
    whenever the relative index is below one.  Miller's algorithm is used
    instead: seed the recurrence well above |z| where psi_n is the decaying
    solution, run it downwards, then fix the arbitrary scale against whichever
    of psi_0 or psi_1 is larger.  sin(z) and cos(z) cannot both be small, so one
    of those two seeds is always well conditioned.

    Args:
        z: function argument (the refractive index times the size parameter)
        nstop: highest order needed

    Returns:
        Array of psi_k(z) for k=0 to nstop
    """
    abs_z = abs(z)
    n_start = int(max(float(nstop), abs_z)) + 25 + int(1.5 * np.sqrt(abs_z))

    psi = np.zeros(n_start + 2, dtype=np.complex128)
    psi[n_start] = 1e-50
    for n in range(n_start, 0, -1):
        psi[n - 1] = (2 * n + 1) / z * psi[n] - psi[n + 1]
        if abs(psi[n - 1]) > 1e200:  # keep the growing tail in range
            for k in range(n - 1, n_start + 2):
                psi[k] /= 1e200

    sin_z = np.sin(z)
    psi_0 = sin_z
    psi_1 = sin_z / z - np.cos(z)
    if abs(psi_1) > abs(psi_0):
        scale = psi_1 / psi[1]
    else:
        scale = psi_0 / psi[0]
    for k in range(n_start + 2):
        psi[k] *= scale

    return psi[: nstop + 1]


@njit((complex128, int64), cache=True)
def _D_calc_down_nb(z, N):
    """
    Compute D_1(z)..D_N(z) using only the downwards recurrence.

    ``_D_calc_nb`` picks between the upwards and downwards recurrences with
    Wiscombe's criterion, which considers the refractive index but not the number
    of terms.  The upwards recurrence loses accuracy once N exceeds |z| (about
    4 digits for a small sphere), and the resulting noise differs between the
    numba and pure-python backends.  The internal-field coefficients are not on
    the hot path, so they always take the stable route.

    Args:
        z: function argument (already multiplied by the refractive index)
        N: highest order needed

    Returns:
        Array of logarithmic derivatives D_k(z) for k=1 to N
    """
    D = np.zeros(N + 1, dtype=np.complex128)
    _D_downwards(z, N, D)
    return D[1:]


@njit((complex128, float64, int64), cache=True)
def _D_calc_nb(m, x, N):
    """
    Compute the logarithmic derivative of ψ_n(z) using the downwards recurrence.

    D_n(z) = d[log ψ_n(z)] = ψ_n'(z)/ψ_n(z)

    here ψ_n(z) is the Riccati-Bessel function of the first kind ψ_n(z)=z*j_n(z)
    were j_n(z) is the spherical Bessel function of order n.

    The zero-based array, D[:], is shifted so that D[0] = D₁(z) = ψ₁'(z)/ψ₁(z)

    Wiscombe's criterion used to pick between the upwards and downwards
    recurrences from the refractive index alone.  The upwards recurrence is
    contaminated once N passes |mx|, which is the case for every small sphere,
    and it left the Mie coefficients wrong by as much as 1% near m=1.  The
    downwards recurrence is stable everywhere, and the criterion already chose it
    for most large spheres, so always taking it costs little.  ``_D_upwards`` is
    kept for comparison and testing.

    Args:
        m: the np.complex128 index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function

    Returns:
        Array of logarithmic derivatives D_k(z) for k=1 to N-1.
    """
    return _D_calc_down_nb(np.complex128(m * x), N)


@njit((complex128, float64, int64), cache=True)
def _an_bn_nb(m, x, n_pole):
    """
    Compute arrays of Mie coefficients a_n and b_n for a sphere.

    When n_pole=0, the routine estimates the size of the arrays based on Wiscombe's
    formula. The length of the arrays is chosen so that the error when the series
    is summed is around 1e-6.

    If n_pole>0, then the array sizes will be n_pole+1. This is useful when
    trying to isolate the behavior of a particular multipole.

    To support resonance calculations, one can specify the number of terms
    to be calculated.  In general, using too few or too many terms increases the
    error rate.  So if you specify the number of terms be aware that you are
    playing with fire.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: the number of An and Bn terms (0 does autosizing)

    Returns:
        a, b: arrays of Mie coefficents An and Bn
    """
    # make sure sign of imaginary part is negative
    m_re = m.real
    m_im = m.imag
    if m_im > 0.0:
        m_im = -m_im
    m = complex(m_re, m_im)

    # make sure array size is large enough
    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    a = np.zeros(nstop, dtype=np.complex128)
    b = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return a, b

    psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - np.cos(x)
    xi_nm1 = np.complex128(psi_nm1 + 1j * np.cos(x))
    xi_n = np.complex128(psi_n + 1j * (np.cos(x) / x + np.sin(x)))

    if m.real > 0.0:
        D = _D_calc_nb(m, x, nstop + 1)

        for n in range(1, nstop):
            temp = D[n - 1] / m + n / x
            a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n - 1] * m + n / x
            b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            psi = (2 * n + 1) * psi_n / x - psi_nm1
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = psi

    else:
        for n in range(1, nstop):
            a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b[n - 1] = psi_n / xi_n
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    if n_pole != 0:
        a = a[:-1]
        b = b[:-1]

    return np.conjugate(a), np.conjugate(b)


# No kernel in this module uses fastmath.  It implies LLVM's nnan and ninf, which
# folded the np.isinf guard below to False and let an infinite index divide by
# zero, and its reassociation also broke the exact Re(a_n) == |a_n|**2 identity
# that a lossless sphere has to satisfy.  Do not add it back.
@njit((complex128, float64, int64))
def _cn_dn_nb(m, x, n_pole):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (np.complex128): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    # make sure sign of imaginary part is negative
    m_re = m.real
    m_im = m.imag
    if m_im > 0.0:
        m_im = -m_im
    m = complex(m_re, m_im)
    mx = m * x

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    c = np.zeros(nstop, dtype=np.complex128)
    d = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return c, d

    # no need to calculate anything when sphere is perfectly conducting
    # (m.real <= 0 or an infinite index): there is no internal field, so c and d
    # stay zero.  The `and` chain must not be mixed with `or` here, otherwise the
    # test is true for every finite index and m=0 divides by zero below.
    if m.real > 0.0 and not np.isinf(m.real) and not np.isinf(m.imag):
        sin_x = np.sin(x)
        cos_x = np.cos(x)

        # xi is the growing solution, so seeding it and running upwards is stable
        xi_nm1 = np.complex128(sin_x + 1j * cos_x)
        xi_n = np.complex128((sin_x / x - cos_x) + 1j * (cos_x / x + sin_x))

        psi_x = _psi_downwards_nb(np.complex128(x), nstop + 1)
        psi_mx = _psi_downwards_nb(np.complex128(mx), nstop + 1)
        Dmx = _D_calc_down_nb(np.complex128(mx), nstop + 1)
        Dx = _D_calc_down_nb(np.complex128(x), nstop + 1)

        for n in range(1, nstop + 1):
            common = (psi_x[n] / psi_mx[n]) * ((Dx[n - 1] + n / x) * xi_n - xi_nm1)

            c[n - 1] = m * common / ((m * Dmx[n - 1] + n / x) * xi_n - xi_nm1)
            d[n - 1] = common / ((Dmx[n - 1] / m + n / x) * xi_n - xi_nm1)

            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi

    if n_pole != 0:
        c = c[:-1]
        d = d[:-1]
    return np.conjugate(c), np.conjugate(d)


@njit((float64, float64[:], float64[:]), cache=True)
def _pi_tau_nb(mu, pi, tau):
    """
    Compute the Mie scattering functions π_n and τ_n for given cosine angles.

    This function fills the pre-allocated arrays `pi` and `tau` with values
    of the Mie scattering functions for a given `mu = cos𝜃`. The function
    uses the recurrence relations for the associated Legendre polynomials
    of the first kind P_n^1. The recurrence relations ensure numerical stability
    and avoids calling scipi.special.lpmv(1, n, cos𝜃) for each n.

    `pi` and `tau` are **zero-based** arrays and therefore

    `pi[n-1]` = 𝜋_n(cos𝜃) = P_n^1(cos𝜃) / sin𝜃

    `tau[n-1]` = 𝜏_n(cos𝜃) = d/d𝜃 P_n^1(cos𝜃)`.

    Args:
        mu (float): The cosine of the scattering angle, `cos(𝜃)`.
        pi (numpy.ndarray): A pre-allocated array to store `pi_n` values.
        tau (numpy.ndarray): A pre-allocated array to store `tau_n` values.

    Returns:
        nothing.  pi and tau are modified
    """
    n_terms = len(pi)
    pi_nm2 = 0
    pi[0] = 1
    for n in range(1, n_terms):
        tau[n - 1] = n * mu * pi[n - 1] - (n + 1) * pi_nm2
        temp = pi[n - 1]
        pi[n] = ((2 * n + 1) * mu * temp - (n + 1) * pi_nm2) / n
        pi_nm2 = temp


@njit((complex128, float64, float64[:], int64), cache=True)
def _S1_S2_nb(m, x, mu, n_pole):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: array of angles, cos(theta), to calculate scattering amplitudes
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a, b = _an_bn_nb(m, x, 0)
    N = len(a)
    if n_pole < 0 or n_pole >= N:
        raise ValueError("n_pole must be 0 (all terms) or a multipole order in 1.." + str(N - 1))

    pi = np.zeros(N)
    tau = np.zeros(N)
    n = np.arange(1, N + 1, dtype=np.float64)
    scale = (2.0 * n + 1.0) / ((n + 1.0) * n)

    nangles = len(mu)
    S1 = np.zeros(nangles, dtype=np.complex128)
    S2 = np.zeros(nangles, dtype=np.complex128)

    # zero-based arrays: multipole order n lives at index n-1
    j = n_pole - 1

    for k in range(nangles):
        _pi_tau_nb(mu[k], pi, tau)
        if n_pole == 0:
            s1 = 0.0 + 0.0j
            s2 = 0.0 + 0.0j
            for i in range(N):
                si = scale[i]
                s1 += si * (pi[i] * a[i] + tau[i] * b[i])
                s2 += si * (tau[i] * a[i] + pi[i] * b[i])
            S1[k] = s1
            S2[k] = s2
        else:
            S1[k] = scale[j] * (pi[j] * a[j] + tau[j] * b[j])
            S2[k] = scale[j] * (tau[j] * a[j] + pi[j] * b[j])

    return np.conjugate(S1), np.conjugate(S2)


@njit((complex128, float64), cache=True)
def _small_conducting_sphere_nb(_m, x):
    """
    Calculate the efficiencies for a small conducting spheres.

    Typically used for small conducting spheres where x < 0.1 and
    m.real == 0

    Args:
        _m: the complex index of refraction of the sphere (unused)
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ahat1 = complex(0, 2.0 / 3.0 * (1 - 0.2 * x**2))
    ahat1 /= complex(1 - 0.5 * x**2, 2.0 / 3.0 * x**3)

    bhat1 = complex(0.0, (x**2 - 10.0) / 30.0)
    bhat1 /= complex(1 + 0.5 * x**2, -(x**3) / 3.0)
    ahat2 = complex(0.0, x**2 / 30.0)
    bhat2 = complex(0.0, -(x**2) / 45.0)

    qsca = x**4 * (6 * np.abs(ahat1) ** 2 + 6 * np.abs(bhat1) ** 2 + 10 * np.abs(ahat2) ** 2 + 10 * np.abs(bhat2) ** 2)
    qext = qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0 / 9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6 * x**4 / qsca

    qback = 9 * x**4 * np.abs(ahat1 - bhat1 - 5 / 3 * (ahat2 - bhat2)) ** 2

    return qext, qsca, qback, g


@njit((complex128, float64), cache=True)
def _small_sphere_nb(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Typically used for small spheres where x<0.1

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    m2 = m * m
    x2 = x * x

    D = m2 + 2 + (1 - 0.7 * m2) * x2
    D -= (8 * m**4 - 385 * m2 + 350) * x**4 / 1400.0
    D += 2j * (m2 - 1) * x**3 * (1 - 0.1 * x2) / 3
    ahat1 = 2j * (m2 - 1) / 3 * (1 - 0.1 * x2 + (4 * m2 + 5) * x**4 / 1400) / D

    bhat1 = 1j * x2 * (m2 - 1) / 45 * (1 + (2 * m2 - 5) / 70 * x2)
    bhat1 /= 1 - (2 * m2 - 5) / 30 * x2

    ahat2 = 1j * x2 * (m2 - 1) / 15 * (1 - x2 / 14)
    ahat2 /= 2 * m2 + 3 - (2 * m2 - 7) / 14 * x2

    T = np.abs(ahat1) ** 2 + np.abs(bhat1) ** 2 + 5 / 3 * np.abs(ahat2) ** 2
    temp = ahat2 + bhat1
    g = (ahat1 * temp.conjugate()).real / T

    qsca = 6 * x**4 * T

    if m.imag == 0:
        qext = qsca
    else:
        qext = 6 * x * (ahat1 + bhat1 + 5 * ahat2 / 3).real

    sback = 1.5 * x**3 * (ahat1 - bhat1 - 5 * ahat2 / 3)
    qback = 4 * np.abs(sback) ** 2 / x2

    return qext, qsca, qback, g


@njit((complex128, float64, int64, boolean), cache=True)
def _single_sphere_nb(m, x, n_pole, e_field):
    """
    Calculate the efficiencies for a single sphere (both m and x are scalars).

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: a non-zero value returns the contribution by the n_pole multipole
        e_field: selects which multipole of order n_pole contributes, the
            electric one a_n (True) or the magnetic one b_n (False).
            Ignored when n_pole == 0.

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0, 0, 0, 0

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_sphere_nb(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_sphere_nb(m, x)

    # sometimes m=0 is used to signal perfectly conducting sphere
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        m = 1 - 10000j

    a, b = _an_bn_nb(m, x, n_pole)
    x2 = x * x

    if n_pole == 0:
        n_terms = len(a)
        qext_acc = 0.0
        qsca_acc = 0.0
        qback_acc = 0.0 + 0.0j
        g_acc = 0.0

        for i in range(n_terms):
            ni = i + 1
            cn = 2.0 * ni + 1.0

            ai = a[i]
            bi = b[i]
            ai_re = ai.real
            ai_im = ai.imag
            bi_re = bi.real
            bi_im = bi.imag

            qext_acc += cn * (ai_re + bi_re)

            if m.imag != 0.0:
                qsca_acc += cn * (ai_re * ai_re + ai_im * ai_im + bi_re * bi_re + bi_im * bi_im)

            # (-1)^n with n starting from 1.
            sign = -1.0 if (ni % 2) == 1 else 1.0
            qback_acc += sign * cn * (ai - bi)

            if i < n_terms - 1:
                aip1 = a[i + 1]
                bip1 = b[i + 1]
                c1n = ni * (ni + 2.0) / (ni + 1.0)
                c2n = cn / ni / (ni + 1.0)
                g_acc += c1n * (ai * np.conjugate(aip1) + bi * np.conjugate(bip1)).real
                g_acc += c2n * (ai * np.conjugate(bi)).real

        qext = 2.0 * qext_acc / x2

        if m.imag == 0.0:
            qsca = qext
        else:
            qsca = 2.0 * qsca_acc / x2

        qback = np.abs(qback_acc) ** 2 / x2
        g = 4.0 * g_acc / qsca / x2

    else:
        # isolate one multipole: the electric term a_n or the magnetic term b_n
        cn = 2.0 * n_pole + 1
        coeff = a[-1] if e_field else b[-1]
        qext = 2.0 * cn * coeff.real / x2
        # the (-1)**n_pole and the sign of b_n drop out of the modulus
        qback = np.abs(cn * coeff) ** 2 / x2
        qsca = qext
        if m.imag < 0:
            qsca = 2.0 * cn * np.abs(coeff) ** 2 / x2

        # pi_n**2 and tau_n**2 are both even in mu, so an isolated multipole of
        # one parity scatters symmetrically about 90 degrees and <cos(theta)>=0
        g = 0.0

    return qext, qsca, qback, g
