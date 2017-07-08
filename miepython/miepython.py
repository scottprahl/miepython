from __future__ import division
import numpy as np

# compute a logarithmic derivative using continued fractions
def Lentz_Dn(z, nstop):

    zinv     =  2.0/z
    alpha    =  (nstop+0.5) * zinv
    aj       = -(nstop+1.5) * zinv
    alpha_j1 = aj+1/alpha
    alpha_j2 = aj
    ratio    = alpha_j1/alpha_j2
    runratio = alpha*ratio

    while abs(abs(ratio)-1.0) > 1e-12 :
        aj = zinv - aj
        alpha_j1 = 1.0/alpha_j1 + aj
        alpha_j2 = 1.0/alpha_j2 + aj
        ratio = alpha_j1/alpha_j2
        zinv  *= -1
        runratio = ratio*runratio

    return -nstop/z+runratio

# downward recurrence
def Dn_downwards(z, nstop):
    D = np.zeros(nstop, dtype=complex)
    temp = Lentz_Dn(z, nstop)
    for n in range(nstop,0,-1) :
        temp =  n/z - 1.0/(temp+n/z)
        D[n-1] = temp
    return D

# upward recurrence
def Dn_upwards(z, nstop):
    D = np.zeros(nstop, dtype=complex)
    D[0] = 0 #unused
    exp = np.exp(-2j*z)
    D[1] = -1/z + (1-exp)/((1-exp)/z-1j*(1+exp))
    for n in range(2,nstop) :
        D[n] = 1/(n/z-D[n-1])-n/z
    return D

# return array of nstop logarithmic derivatives
def Dn_calc(m, x, nstop):
    z = m * x
    if abs(z.imag) > 13.78*m.real**2 - 10.8*m.real + 3.9 :
        return Dn_upwards(z, nstop)
    else :
        return Dn_downwards(z, nstop)

# calculate coefficients An & Bn needed for mie calculations
def mie_An_Bn(m,x):

    nstop = int(x + 4.05 * x**0.33333 + 2.0)+1

    if m.real > 0.0 :
        D = Dn_calc(m, x, nstop+1)

    a = np.zeros(nstop-1, dtype=complex)
    b = np.zeros(nstop-1, dtype=complex)

    psi_nm1 = np.sin(x)                   # nm1 = n-1 = 0
    psi_n   = psi_nm1/x - np.cos(x)       # n=1
    xi_nm1  = complex(psi_nm1, np.cos(x))
    xi_n    = complex(psi_n,   np.cos(x)/x+np.sin(x))

    for n in range(1,nstop):
        if m.real==0.0 :
            a[n-1] = (n*psi_n/x - psi_nm1)/(n*xi_n/x-xi_nm1)
            b[n-1] = psi_n/ xi_n
        else :
            temp = D[n]/m+n/x
            a[n-1] = (temp*psi_n-psi_nm1)/(temp*xi_n-xi_nm1)
            temp = D[n]*m+n/x
            b[n-1] = (temp*psi_n-psi_nm1)/(temp*xi_n-xi_nm1)

        xi   = (2*n+1)*xi_n/x - xi_nm1
        xi_nm1  = xi_n
        xi_n  = xi
        psi_nm1 = psi_n
        psi_n = xi_n.real

    return [a,b]

def small_conducting_mie(m,x):

    ahat1 = complex(0.0,2.0/3.0*(1.0-0.2*x**2))/complex(1.0-0.5*x**2,2.0/3.0*x**3)
    bhat1 = complex(0.0,(x**2-10.0)/30.0)/complex(1+0.5*x**2,-x**3/3.0)
    ahat2 = complex(0.0, x**2/30.)
    bhat2 = complex(0.0,-x**2/45.)

    qsca = x**4*(6*abs(ahat1)**2 + 6*abs(bhat1)**2 + 10*abs(ahat2)**2 + 10*abs(bhat2)**2)
    qext = qsca
    qabs = 0
    g =  ahat1.imag * (ahat2.imag+bhat1.imag)
    g += bhat2.imag * (5.0/9.0*ahat2.imag+bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6*x**4/qsca

    qback1 = ahat1.real-bhat1.real
    qback2 = ahat1.imag-bhat1.imag-(5.0/3.0)*(ahat2.imag+bhat2.imag)
    qback = 6*x**2*(qback1**2+qback2**2)

    return [qext, qsca, qabs, qback, g]

def small_mie(m,x):
    m2 = m * m
    m4 = m2 * m2
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    D=m2+2+(1-0.7*m2)*x2-(8*m4-385*m2+350)*x4/1400.0 + 2j*(m2-1)*x3*(1-0.1*x2)/3
    ahat1 = 2j*(m2-1)/3*(1-0.1*x2+(4*m2+5)*x4/1400)/D
    bhat1 = 1j*x2*(m2-1)/45 * (1+(2*m2-5)/70*x2)/(1-(2*m2-5)/30*x2)
    ahat2 = 1j*x2*(m2-1)/15 * (1-x2/14)/(2*m2+3-(2*m2-7)/14*x2)

    qext = 6*x*(ahat1+bhat1+5*ahat2/3).real
    T = abs(ahat1)**2+abs(bhat1)**2+5/3*abs(ahat2)**2
    temp = ahat2+bhat1
    g = (ahat1*temp.conjugate()).real/T

    qabs = 0
    qsca = qext
    if m.imag < 0 :
        qsca = 6*x4*T
        qabs = qext-qsca

    qback = 2.25*x4*abs(ahat1-bhat1-5*ahat2/3)**2
    return [qext, qsca, qabs, qback, g]

# return list of mie efficiencies and scattering anisotropy g
def mie(m, x):
    if m.real==0 and x < 0.1 :
        small_conducting_mie(m,x)

    if m.real>0.0 and abs(m) * x < 0.1 :
        return small_mie(m,x)

    a,b = mie_An_Bn(m,x)
    print(a)

    nmax = len(a)
    n    = np.arange(1,nmax+1)
    cn   = 2.0*n + 1.0
    x2   = x*x

    qext = 2*np.sum(cn * (a.real + b.real))/x2
    qabs = 0
    qsca = qext

    if m.imag != 0:
        qsca = 2*np.sum(cn*(abs(a)**2 + abs(b)**2))/x2
        qabs = qext-qsca

    qback = abs(np.sum( (-1)**n * cn * (a - b) ))**2/x2

    c1n  = n*(n + 2)/(n + 1)
    c2n  = cn/n/(n + 1)
    g=0
    for i in range(nmax-1):
        asy1 = c1n[i] * (a[i] * a[i+1].conjugate() + b[i] * b[i+1].conjugate()).real
        asy2 = c2n[i] * (a[i] * b[i].conjugate()).real
        g += 4*(asy1 + asy2)/qsca/x2

    return [qext, qsca, qabs, qback, g]

# calculate S1 and S2 arrays needed to find scattering function
def mie_S1_S2(m,x,mu):
    nangles = len(mu)

    a,b = mie_An_Bn(m,x)
    nstop = len(a)

    S1 = np.zeros(nangles, dtype=complex)
    S2 = np.zeros(nangles, dtype=complex)

    for k in range(nangles):
        pi_nm2 = 0
        pi_nm1 = 1
        for n in range(1,nstop):
            tau_nm1 =  n * mu[k] * pi_nm1 - (n + 1) * pi_nm2
            S1[k] += (2*n+1)*(pi_nm1 * a[n-1] + tau_nm1 * b[n-1])/(n+1)/n
            S2[k] += (2*n+1)*(tau_nm1 * a[n-1] + pi_nm1 * b[n-1])/(n+1)/n

            temp = pi_nm1
            pi_nm1 = ((2*n+1) * mu[k] * pi_nm1 - (n+1) * pi_nm2)/n
            pi_nm2 = temp

    return [S1,S2]
