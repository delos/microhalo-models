import numpy as np
from scipy.integrate import quad
from numba import njit

# stellar IMFs
def generate_coef_imf(a_imf,m_imf):
  coefs = np.zeros_like(a_imf)
  if len(a_imf) == 1:
    coefs[0] = 1
  elif len(a_imf) > 1:
    coefs[0] = m_imf[1]**a_imf[0]
    coefs[1] = m_imf[1]**a_imf[1]
  if len(a_imf) > 2:
    coefs[2] = (m_imf[2]/m_imf[1])**-a_imf[1]*m_imf[2]**a_imf[2]
  if len(a_imf) > 3:
    coefs[3] = (m_imf[2]/m_imf[1])**-a_imf[1]*(m_imf[3]/m_imf[2])**-a_imf[2]*m_imf[3]**a_imf[3]
  return coefs
    
def generate_int_imf(a_imf,m_imf,coefs):
  ints = np.zeros_like(a_imf)
  for i in range(len(a_imf)-1):
    ints[i] = coefs[i]/(1-a_imf[i])*(m_imf[i+1]**(1-a_imf[i])-m_imf[i]**(1-a_imf[i]))
  i = len(a_imf)-1
  ints[i] = coefs[i]/(1-a_imf[i])*(-m_imf[i]**(1-a_imf[i]))
  return ints

def generate_imf(a_imf,m_imf):
  coefs = generate_coef_imf(a_imf,m_imf)
  ints = generate_int_imf(a_imf,m_imf,coefs)
  integral = np.sum(ints)
  return a_imf,m_imf,integral,coefs,ints

@njit
def xi(m,a_imf,m_imf,integral,coefs,ints):
  for i in range(len(a_imf)-1,-1,-1):
    if m > m_imf[i]:
      return coefs[i]*m**-a_imf[i] / integral
  return 0

@njit
def xic(m,a_imf,m_imf,integral,coefs,ints):
  for i in range(len(a_imf)-1,-1,-1):
    if m > m_imf[i]:
      return (np.sum(ints[:i]) + coefs[i]/(1-a_imf[i])*(m**(1-a_imf[i])-m_imf[i]**(1-a_imf[i]))) / integral
  return 0

@njit
def xicinv(p,a_imf,m_imf,integral,coefs,ints):
  imfsize = a_imf.size
  pi = p*integral
  for i in range(imfsize):
    m0 = ((1-a_imf[i])*(coefs[i]/(1-a_imf[i])*m_imf[i]**(1-a_imf[i])+pi-np.sum(ints[:i]))/coefs[i])**(1./(1-a_imf[i]))
    if i+1 == imfsize or m0 < m_imf[i+1]:
      return m0
  return 0

@njit
def xicinv_vec(p,a_imf,m_imf,integral,coefs,ints):
  imfsize = a_imf.size
  pi_list = p*integral
  ret = np.zeros_like(pi_list)
  for i in range(imfsize):
    for j,pi in enumerate(pi_list):
      m0 = ((1-a_imf[i])*(coefs[i]/(1-a_imf[i])*m_imf[i]**(1-a_imf[i])+pi-np.sum(ints[:i]))/coefs[i])**(1./(1-a_imf[i]))
      if (i+1 == imfsize or m0 < m_imf[i+1]) and ret[j]==0.:
        ret[j] = m0
  return ret

def Mstar_mean(a_imf,m_imf,integral,coefs,ints):
  integrand = lambda m: xi(m,a_imf,m_imf,integral,coefs,ints) * m
  return quad(integrand,m_imf[0],np.inf)[0]