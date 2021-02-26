import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator

data = np.load('bulge-sym.npz')
RXY_ = data['RXY']
z_ = data['z']
rho_ = data['density']

rho_interp = RegularGridInterpolator((RXY_,z_),rho_,fill_value=0,bounds_error=False)

csqrt = njit(lambda x: np.sqrt(x))

def rho(R,z):
  RXY = csqrt(R**2-z**2)
  return rho_interp((RXY,np.abs(z)))

def rho_vec(R,z):
  RXY = np.sqrt(R**2-z**2)
  return rho_interp((RXY,np.abs(z)))
