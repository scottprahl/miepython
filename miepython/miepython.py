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

# return array of nstop logarithmic derivatives
def Dn_calc(m, x, nstop):
    
    D = np.zeros(nstop, dtype=complex)
    z = m * x
    
    if abs(z.imag) < 13.78*m.real**2 - 10.8*m.real + 3.9 :
        # upward recurrence
        D[0] = np.cos(z)/np.sin(z)
        for n in range(1,nstop) :
            D[n] = 1/(n/z-D[n-1])-n/z

    else :
        # downward recurrence
        D[nstop-1] = Lentz_Dn(z, nstop)
        for n in range(nstop-1,1,-1) :
            D[n-1] = n/z - 1.0/(D[n]+n/z)

    return D
    
# calculate coefficients An & Bn needed for Mie calculations
def Mie_An_Bn(m,x):

    nstop = int(x + 4.05 * x**0.33333 + 2.0)+1
 
    D = Dn_calc(m, x, nstop)
    a = np.zeros(nstop-1, dtype=complex)
    b = np.zeros(nstop-1, dtype=complex)

    psi_nm1 = np.sin(x)                   # nm1 = n-1 = 0
    psi_n   = psi_nm1/x - np.cos(x)       # n=1
    xi_nm1  = complex(psi_nm1, np.cos(x))
    xi_n    = complex(psi_n,   np.cos(x)/x+np.sin(x))

    for n in range(1,nstop): 
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
    
# return list of Mie efficiencies and scattering anisotropy g
def mie(m, x):
    
    a,b = Mie_An_Bn(m,x)    

    nmax = len(a)
    n    = np.arange(1,nmax+1)
    cn   = 2*n + 1
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
def Mie_S1_S2(m,x,mu_array):  
    nangles = len(mu)
    
    a,b = Mie_An_Bn(m,x)
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
