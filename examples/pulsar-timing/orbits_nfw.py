import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq

import profiles_nfw as profiles
from halo_structure_nfw import KEcirc, Phi, mass_profile, Rfromr, EfromvE, vfromRvE

# orbits
@njit
def E_l_from_rc_eta(rc,eta,Rhos,Rs,G):
  E = KEcirc(rc,Rhos,Rs,G) + Phi(rc,Rhos,Rs,G)
  l = eta*(G*mass_profile(rc,Rhos,Rs)*rc)**.5
  return E,l
def initialize_rc_E(Rhos,Rs,G,n=10000,xmax=1e5):
  list_rc_E = np.concatenate((np.linspace(0,Rs,n//10)[:-1],np.geomspace(Rs,xmax*Rs,(9*n)//10)))
  list_E_rc = np.zeros_like(list_rc_E)
  for i,rc in enumerate(list_rc_E):
    list_E_rc[i] = KEcirc(rc,Rhos,Rs,G) + Phi(rc,Rhos,Rs,G)
  return interp1d(list_E_rc,list_rc_E,bounds_error=False)
#rc_from_E = initialize_rc_E()
@njit
def etafun(rc,E,l,Rhos,Rs,G):
  if rc > 0:
    return np.divide(l,rc*np.sqrt(2*(E-Phi(rc,Rhos,Rs,G))))
  return 0
def rc_eta_from_E_l(E,l,Rhos,Rs,G,rc_from_E):
  rc = rc_from_E(E)
  eta = etafun(rc,E,l,Rhos,Rs,G)
  return rc,eta
def P(R,vE,Rc=None):
  pr = profiles.potential(R)
  with np.errstate(divide='ignore',invalid='ignore'):
    if Rc is None:
      return np.where((pr>vE)&(vE<1)&(vE>0),R**2*np.power(pr-vE,.5,where=pr>vE)*profiles.F(vE),0)
    else:
      pc = profiles.potential(Rc)
      pr -= pc
      return np.where((pr>vE)&(vE>0),R**2*np.sqrt(pr-vE,where=pr>vE)*profiles.F(vE+pc),0)
def randrv(r,v,N,R0,Rhos,Rs,G):
  rc_from_E = initialize_rc_E(Rhos,Rs,G)
  vrelmax = np.sqrt(-2*Phi(R0,Rhos,Rs,G))+np.abs(v)
  Ngen = 0
  rc = []
  eta = []
  mu = []
  phi = []
  vrel = []
  R = Rfromr(r,Rs)
  while(Ngen < N):
    vE_ = np.random.rand(N)
    Psamp = np.random.rand(N)*.07
    probs = np.vectorize(P)(R,vE_)
    idx = Psamp < probs
    if np.sum(idx) == 0:
      continue
    E_ = EfromvE(vE_[idx],Rhos,Rs,G)
    v_ = vfromRvE(R,vE_[idx],Rhos,Rs,G)
    mu_ = 2*np.random.rand(len(E_))-1 # = cos(angle from radius)
    phi_ = np.random.rand(len(E_))*2*np.pi # angle to sun's velocity within tangent plane
    # <sqrt(1-mu^2)cosphi, sqrt(1-mu^2)sinphi, mu>.<1,0,0> = sqrt(1-mu^2)cosphi
    proj_ = np.sqrt(1-mu_**2)*np.cos(phi_)
    vrel_ = np.abs(v_*proj_-v)
    vsamp = np.random.rand(len(E_))*vrelmax
    L_ = v_*r*np.sqrt(1-mu_**2)
    rc_,eta_ = np.vectorize(lambda E,l: rc_eta_from_E_l(E,l,Rhos,Rs,G,rc_from_E))(E_,L_)
    idx = np.isfinite(rc_)&(vsamp<vrel_)
    rc = np.concatenate((rc,rc_[idx]))
    eta = np.concatenate((eta,eta_[idx]))
    mu = np.concatenate((mu,mu_[idx]))
    phi = np.concatenate((phi,phi_[idx]))
    vrel = np.concatenate((vrel,vrel_[idx]))
    Ngen = len(rc)
  # scale vrel by np.sqrt(1-(2*np.random.rand()-1)**2) -> vrel_perp
  return rc[:N],eta[:N],mu[:N],phi[:N],vrel[:N]
@njit
def theta_integrand(r,E,l,Rhos,Rs,G):
  r2rdot2 = 2*E*r**2 - l**2 - 2*Phi(r,Rhos,Rs,G)*r**2
  if r2rdot2 <= 0:
    return 0
  return 1./(r*np.sqrt(r2rdot2))
@njit
def time_integrand(r,E,l,Rhos,Rs,G):
  r2rdot2 = 2*E*r**2 - l**2 - 2*Phi(r,Rhos,Rs,G)*r**2
  if r2rdot2 <= 0:
    return 0
  return r/np.sqrt(r2rdot2)
@njit
def posfun(t,time_shift,halfperiod,halfangle):
  t += time_shift
  i_halfperiod = np.floor(t/halfperiod)
  t_halfperiod = t - halfperiod * i_halfperiod # 0 to halfperiod
  theta0 = halfangle * i_halfperiod
  if i_halfperiod % 2 == 1:
    t_halfperiod = halfperiod - t_halfperiod # halfperiod to 0
    theta0 += halfangle
    if t_halfperiod < 0:
      t_halfperiod = 0
    if t_halfperiod > halfperiod:
      t_halfperiod = halfperiod
    return t_halfperiod,theta0,-1
  else:
    if t_halfperiod < 0:
      t_halfperiod = 0
    if t_halfperiod > halfperiod:
      t_halfperiod = halfperiod
    return t_halfperiod,theta0,1
@njit
def velfun(r,theta,r1,r2,E,l,Rhos,Rs,G):
    vr = 2*(E-Phi(r,Rhos,Rs,G))-(l/r)**2
    if vr <= 0:
      vr = 0
    vt = l/r
    return np.sqrt(vr),vt

csin = njit(lambda x: np.sin(x))
ccos = njit(lambda x: np.cos(x))
ctan = njit(lambda x: np.tan(x))
csqrt = njit(lambda x: np.sqrt(x))
clog = njit(lambda x: np.log(x))

class Orbit(object):
    
  def init_theta(self):
    # angles
    integrand = lambda r: theta_integrand(r,self.E,self.l,self.Rhos,self.Rs,self.G)
    self.orbit_theta_table = np.zeros_like(self.orbit_r_table)
    for i,r in enumerate(self.orbit_r_table):
      if i>0:
        rprev = self.orbit_r_table[i-1]
      else:
        rprev = self.r1
      result = self.l*quad(integrand,rprev,r)[0]
      self.orbit_theta_table[i] = self.orbit_theta_table[i-1] + result
    self.halfangle = self.orbit_theta_table[-1]-self.orbit_theta_table[0]
    self.angle = 2*self.halfangle
  
  def init_time(self):
    # times
    integrand = lambda r: time_integrand(r,self.E,self.l,self.Rhos,self.Rs,self.G)
    self.orbit_time_table = np.zeros_like(self.orbit_r_table)
    for i,r in enumerate(self.orbit_r_table):
      if i>0:
        rprev = self.orbit_r_table[i-1]
      else:
        rprev = self.r1
      result = quad(integrand,rprev,r)[0]
      self.orbit_time_table[i] = self.orbit_time_table[i-1] + result
    self.interp_r_t = interp1d(self.orbit_time_table,self.orbit_r_table) # r1 to r2
    self.interp_theta_t = interp1d(self.orbit_time_table,self.orbit_theta_table) # 0 to halfangle
    self.halfperiod = self.orbit_time_table[-1]-self.orbit_time_table[0]
    self.period = 2*self.halfperiod
  
  def __init__(self,rc,eta,Rhos,Rs,G,R0,phi=0,n=10000):
    self.rc = rc
    self.eta = eta
    self.E,self.l = E_l_from_rc_eta(rc,eta,Rhos,Rs,G)
    self.Rhos = Rhos
    self.Rs = Rs
    self.G = G
    
    # radii
    r2rdot2 = lambda r: 2*self.E*r**2 - self.l**2 - 2*Phi(r,Rhos,Rs,G)*r**2
    self.r1 = brentq(r2rdot2,0.,rc)
    self.r2 = brentq(r2rdot2,rc,rc*1e2)
    rdiv = (self.r1*self.r2)**.5
    #orbit_r_table = np.linspace(r1,r2,n)
    self.orbit_r_table = np.concatenate((
        np.geomspace(self.r1-self.r1*.9,rdiv-self.r1*.9,n//2+1)[:-1]+self.r1*.9,
        -(np.geomspace(rdiv-rdiv*.9,self.r2-rdiv*.9,n//2)+rdiv*.9)[::-1]+rdiv+self.r2,
        ))
    
    # angle, time
    self.init_theta()
    self.init_time()
    self.time_shift = float(interp1d(self.orbit_r_table,self.orbit_time_table,
      bounds_error=False,fill_value=(self.orbit_time_table[0],self.orbit_time_table[-1]),)(R0))
    self.theta_shift = float(interp1d(self.orbit_time_table,self.orbit_theta_table,
      bounds_error=False,fill_value=(self.orbit_theta_table[0],self.orbit_theta_table[-1]),)(self.time_shift))
    self.cosphi = np.cos(phi)
    self.sinphi = np.sin(phi)
  
  def pos(self,t):
    t_halfperiod, theta0, mult = posfun(t,self.time_shift,self.halfperiod,self.halfangle)
    theta = mult * self.interp_theta_t(t_halfperiod) + theta0 + self.theta_shift
    r = self.interp_r_t(t_halfperiod)
    return r,theta
  
  def vel(self,r,theta):
    return velfun(r,theta,self.r1,self.r2,self.E,self.l,self.Rhos,self.Rs,self.G)
  
  def z(self,r,theta):
    return r*csin(theta)*self.sinphi