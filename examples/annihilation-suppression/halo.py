import numpy as np
import profiles

class Halo(object):
  def __init__(self,
               r_s,
               rho_s,
               params=[1,3,1], # or [1,3,1.5]
               G=None, # overrides use_h_units and use_km_s if both set
               rhoM=None # overrides use_h_units and use_km_s if both set
               ):
    if params not in profiles.supported_params:
      raise ValueError
    self.params = params
    self.G = G
    self.rho_s = rho_s*1./profiles.density_norm(params)
    self.r_s = r_s
    # conversions
    self.E_unit = 4*np.pi*self.G*self.rho_s*self.r_s**2
    self.m_unit = 4*np.pi*self.r_s**3*self.rho_s
    self.L_unit = (4*np.pi*self.G*self.rho_s*self.r_s**4)**.5
    self.f_unit = (4*np.pi*self.G)**1.5*self.r_s**3*self.rho_s**.5
    self.v_unit = (4*np.pi*self.G*self.rho_s*self.r_s**2)**.5
  
  # dimensionless/dimensionful conversions
  def EfromvE(self,vE):
    return -vE*self.E_unit
  def vEfromE(self,E):
    return -E/self.E_unit
  
  # dimensionful structure functions
  def density_profile(self,r):
    return self.rho_s*profiles.density(r/self.r_s,self.params)
  def mass_profile(self,r):
    return self.m_unit*profiles.mass(r/self.r_s,self.params)
  def massp_profile(self,r): # dM/dlnr
    return 4*np.pi*r**3*self.density_profile(r)
  def density_mean_profile(self,r):
    return self.rho_s*profiles.density_mean(r/self.r_s,self.params)
  def r3_over_m_profile(self,r):
    return 1./(4*np.pi*self.rho_s) * profiles.r3_over_mass(r/self.r_s,self.params)
  def Phi(self,r):
    return self.EfromvE(profiles.potential(r/self.r_s,self.params))
  def dPhidr(self,r):
    return self.EfromvE(profiles.dPsidR(r/self.r_s,self.params))/self.r_s
  def d2Phidr2(self,r):
    return self.EfromvE(profiles.d2PsidR2(r/self.r_s,self.params))/self.r_s**2
  def f(self,E):
    return profiles.F(self.vEfromE(E),self.params)/self.f_unit
  def f_aniso(self,E,L,Ra=1):
    return profiles.F_aniso(self.vEfromE(E),L/self.L_unit,Ra,self.params)/self.f_unit
  def sigma2(self,r):
    return -self.EfromvE(profiles.velocity_dispersion_squared(r/self.r_s,self.params))
  def sigma(self,r):
    return self.sigma2(r)**.5
  def sigmar2(self,r):
    return -self.EfromvE(profiles.velocity_dispersion_radial_squared(r/self.r_s,self.params))
  def sigmar(self,r):
    return self.sigmar2(r)**.5
  def KEcirc(self,r):
    return -self.EfromvE(profiles.KE_circular(r/self.r_s,self.params))
  