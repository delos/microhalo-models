import numpy as np
from scipy.interpolate import interp1d
from numba import njit

data = np.load('fv.npz')
a_ = data['a']
f_ = data['f']

fi_ = interp1d(a_,f_)
fl_ = njit(lambda a: a + 1./a)

def f(a):
  a = np.abs(a)
  if a <= 5:
    return fi_(a)
  return fl_(a)

def f_vec(a):
  a = np.abs(a.astype(float))
  return np.piecewise(a,[a<=5,a>5],[fi_,lambda a: a+1./a])