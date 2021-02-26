import numpy as np
from numba import njit

import profiles_nfw as profiles

# halo structure
@njit
def rfromR(R,Rs):
  return Rs*R
@njit
def Rfromr(r,Rs):
  return r/Rs
@njit
def EfromvE(vE,Rhos,Rs,G):
  return -vE*4*np.pi*G*Rhos*Rs**2
@njit
def vEfromE(E,Rhos,Rs,G):
  return -E/(4*np.pi*G*Rhos*Rs**2)
@njit
def MfromvM(vM,Rhos,Rs):
  return 4*np.pi*Rs**3*Rhos*vM
@njit
def rhofromvrho(vrho,Rhos):
  return Rhos*vrho
@njit
def density_profile(r,Rhos,Rs):
  return rhofromvrho(profiles.density(Rfromr(r,Rs)),Rhos)
@njit
def mass_profile(r,Rhos,Rs):
  return MfromvM(profiles.mass(Rfromr(r,Rs)),Rhos,Rs)
@njit
def massp_profile(r,Rhos,Rs): # dM/dlnr
  return 4*np.pi*r**3*density_profile(r,Rhos,Rs)
@njit
def density_mean_profile(r,Rhos,Rs):
  return rhofromvrho(profiles.density_mean(Rfromr(r,Rs)),Rhos)
@njit
def r3_over_m_profile(r,Rhos,Rs):
  return 1./(4*np.pi*Rhos) * profiles.r3_over_mass(Rfromr(r,Rs))
@njit
def Phi(r,Rhos,Rs,G):
  return EfromvE(profiles.potential(Rfromr(r,Rs)),Rhos,Rs,G)
@njit
def dPhidr(r,Rhos,Rs,G):
  return EfromvE(profiles.dPsidR(Rfromr(r,Rs)),Rhos,Rs,G)/Rs
@njit
def d2Phidr2(r,Rhos,Rs,G):
  return EfromvE(profiles.d2PsidR2(Rfromr(r,Rs)),Rhos,Rs,G)/Rs**2
@njit
def f(E,Rhos,Rs,G):
  return profiles.F(vEfromE(E,Rhos,Rs,G))/((4*np.pi*G)**1.5*Rs**3*Rhos**.5)
@njit
def sigma2(r,Rhos,Rs,G):
  return -EfromvE(profiles.velocity_dispersion_squared(Rfromr(r,Rs)),Rhos,Rs,G)
@njit
def sigma(r,Rhos,Rs,G):
  return np.sqrt(sigma2(r,Rhos,Rs,G))
@njit
def sigmar2(r,Rhos,Rs,G):
  return -EfromvE(profiles.velocity_dispersion_radial_squared(Rfromr(r,Rs)),Rhos,Rs,G)
@njit
def sigmar(r,Rhos,Rs,G):
  return sigmar2(r,Rhos,Rs,G)**.5
@njit
def KEcirc(r,Rhos,Rs,G):
  return -EfromvE(profiles.KE_circular(Rfromr(r,Rs)),Rhos,Rs,G)
@njit
def vfromRvE(R,vE,Rhos,Rs,G,Rc=None):
  if Rc is None:
    V = (2*(profiles.potential(R)-vE))**.5 #vE = Psi-.5*V**2
  else:
    V = (2*(profiles.potential(R)-profiles.potential(Rc)-vE))**.5 #vE = Psi-.5*V**2
  return V*(4*np.pi*G*Rhos*Rs**2)**.5
