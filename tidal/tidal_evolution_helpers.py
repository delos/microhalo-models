import numpy as np

from orbit_functions_nfw import tidefun

from tidal_evolution_parameters import *

# general helpers

afun = lambda r,a,b: a-b*tidefun(r)
brfun = lambda r,a: (1+a*tidefun(r))

# J factor

def scale_J_time(n,x,y,z,B=None):
  b = b_coef*x**-b_index*brfun(y,b_coefr)
  a = a_coefr[0] - a_coefr[1]*tidefun(y)
  c = c_coef*z**c_index
  lnJ = lambda n: np.where(c==1,b*(a-np.log(n)),b*(a-1./(1-c)*(n**(1-c)-1)))
  #dlnJ_dn = -b*n**-c
  #dlnJ_dlnn = -b*n**(1-c)
  n0 = np.where(n>=1,n,1)
  if B is None:
    ret = np.exp(lnJ(n0))
  else:
    n1 = (b/B)**(1./(c-1))
    ret = np.where(n1>1,
        np.where(n0<=n1,np.exp(lnJ(n0)),np.exp(b*(a-1./(1-c)*(B/b-1)))*n1/n0),
        np.exp(a)/n0
        )
  return np.where(n>=1,ret,1-n+n*ret)

def scale_J_radius(r,x,y):
  d = d_coef*x**-d_index
  d = np.where(d>.5,.5,d)
  d *= (1-tidefun(y)/2)
  return 1-d*(r-1)


# rmax

def scale_r_time(n,x,y,z,B=None):
  b = br_coef*x**-br_index*(1+br_coefr*tidefun(y))
  a = ar_coefs[0]*np.log(x/ar_coefs[1])*(1-tidefun(y)*.5) - ar_coefs[2]*tidefun(y)
  c = cr_coef*z**cr_index
  lnr = lambda n: np.where(c==1,b*(a-np.log(n)),b*(a-1./(1-c)*(n**(1-c)-1)))
  n0 = np.where(n>=1,n,1)
  if B is None:
    ret = np.exp(lnr(n0))
  else:
    n1 = (b/B)**(1./(c-1))
    ret = np.where(n1>1,
        np.where(n0<=n1,np.exp(lnr(n0)),np.exp(b*(a-1./(1-c)*(B/b-1)))*n1/n0),
        np.exp(a)/n0
        )
  return np.where(n>=1,ret,1-n+n*ret)

# vmax

def scale_v_time(n,x,y,z,B=None):
  b = bv_coef*x**-bv_index*(1+bv_coefr*tidefun(y))
  a = av_coefs[0]*np.log(x/av_coefs[1])*(1-tidefun(y)*.5) - av_coefs[2]*tidefun(y)
  c = cv_coef*z**cv_index
  lnv = lambda n: np.where(c==1,b*(a-np.log(n)),b*(a-1./(1-c)*(n**(1-c)-1)))
  n0 = np.where(n>=1,n,1)
  if B is None:
    ret = np.exp(lnv(n0))
  else:
    n1 = (b/B)**(1./(c-1))
    ret = np.where(n1>1,
        np.where(n0<=n1,np.exp(lnv(n0)),np.exp(b*(a-1./(1-c)*(B/b-1)))*n1/n0),
        np.exp(a)/n0
        )
  return np.where(n>=1,ret,1-n+n*ret)