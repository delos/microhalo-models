import numpy as np
from scipy.integrate import quad

import ctypes
from scipy.special.cython_special import j1 as cj1, hyp2f1 as chyp2f1
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
j1 = functype(cj1)
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
hyp2f1 = functype(chyp2f1)
from scipy.interpolate import interp1d
from numba import njit

'''
Let's redefine stellar_encounters.postencounter_density_profile(x) here so that
we can speed it up with numba.
'''
alpha = 0.78
beta = 5.
@njit
def profile(x):
  q = (1./3*x**alpha)**beta
  return np.exp(-1./alpha * x**alpha * (1+q)**(1-1/beta) * hyp2f1(1,1,1+1/beta,-q))/x

'''
Compute the window function.
'''
@njit
def W_integrand(r,k):
  return profile(r) * 4*np.pi*r * np.sin(k*r)/k
@njit
def W_integrand_sin(r,k):
  return profile(r) * 4*np.pi*r/k
csin = njit(lambda x: np.sin(x))

ks = np.geomspace(1e-5,1e5,1000)
Ws = np.zeros_like(ks)

for i,k in enumerate(ks):
  if k < 1:
    result = quad(lambda r: W_integrand(r,k),0,np.inf,limit=50000)
  else:
    result = quad(lambda r: W_integrand_sin(r,k),1e-12,np.inf,weight='sin',wvar=k,limit=50000)
  if np.abs(result[1]/result[0]) < 1e-3:
    Ws[i] = result[0]
ks = ks[Ws > 0]
Ws = Ws[Ws > 0]

np.savetxt('W.txt',np.stack((ks,Ws)).T,header='k*r_s, W*M/(rho_s*r_s^3)')

'''
Compute the form factor.
'''
_interp_W = interp1d(ks,Ws)
def W(k):
  if k < ks[0]:
    return Ws[0]
  elif k > ks[-1]:
    return Ws[-1] * (k/ks[-1])**-2 # extrapolate at small radii
  else:
    return _interp_W(k)
  
xs = np.geomspace(1e-5,1e4,300)
Fs = np.zeros_like(xs)

for i,x in enumerate(xs):
  print(i)
  result = quad(lambda k: W(k)*j1(k*x),0,np.inf,limit=100000)
  if np.abs(result[1]/result[0]) < 1e-3:
    Fs[i] = x*result[0]

xs = xs[Fs>0]
Fs = Fs[Fs>0]
np.savetxt('F.txt',np.stack((xs,Fs)).T,header='x/r_s, F*M/(rho_s*r_s^3)')