import numpy as np

def hanning_window(k, kmin, kmax, dk):
    condlist = []
    condlist.append((kmin-dk/2.0 <= k) & (k <  kmin+dk/2.0))
    condlist.append((kmin+dk/2.0 <= k) & (k <= kmax-dk/2.0))
    condlist.append((kmax-dk/2.0 <= k) & (k <= kmax+dk/2.0))

    funclist = []
    funclist.append(lambda x: np.sin(np.pi*(x-kmin+dk/2.0)/(2.0*dk))**2.0)
    funclist.append(1.0)
    funclist.append(lambda x: np.cos(np.pi*(x-kmax+dk/2.0)/(2.0*dk))**2.0)

    return np.piecewise(k, condlist, funclist) 

def xafsft(r_min, r_max, xk, ccpath, kweight=0):
    ccpath *= xk**kweight
    delta_k = 0.05
    nfft = 2048
    cnorm = delta_k * 1.0/np.sqrt(np.pi) 
    chi_r = np.fft.fft(cnorm*ccpath, n=nfft)

    delta_r = np.pi/(nfft*delta_k)                                              
    r = np.array([ i*delta_r for i in xrange(len(chi_r)) ])
    ir_min = np.argmin(np.abs(r_min-r))
    ir_max = np.argmin(np.abs(r_max-r))

    return r[ir_min:ir_max], chi_r[ir_min:ir_max]
