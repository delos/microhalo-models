import numpy as np
from scipy.special import spence
from numba import njit

log2 = np.log(2)
log4 = np.log(4)

@njit
def density(R):
  return 1./(R*(1+R)**2)
@njit
def mass(R):
  if R < 0.1:
    return R**2/2. - 2.*R**3/3. + 3.*R**4/4 - 4*R**5/5 + 5.*R**6/6
  else:
    return np.log(1+R)-R/(1.+R)
@njit
def integrated_density_over_r(R):
  return R/(1.+R)
@njit
def density_mean(R):
  return mass(R)*3./R**3
@njit
def r3_over_mass(R):
  if R < 0.1:
    return 2*R + 8*R**2/3. + 5*R**3/9. - 8*R**4/135. + 17*R**5/810. - 86*R**6/8505.
  else:
    return R**3/mass(R)
@njit
def potential(R):
  if R < 0.1:
    return 1.-R/2.+R**2/3.-R**3/4.+R**4/5.-R**5/6.+R**6/7.
  else:
    return np.divide(np.log(1+R),R)
@njit
def velocity_dispersion_radial_squared(R):
  logR = np.log(R)
  if R < 0.1:
    return .25*(-23+2*np.pi**2-2*logR)*R+(-59./6+np.pi**2-logR)*R**2+1./24*(-101+12*np.pi**2-12*logR)*R**3+(11*R**4)/60.-(13*R**5)/240.+(37*R**6)/1400.
  elif R > 10.:
    return np.divide(-3./16+logR/4,R) + np.divide(69./200+logR/10,R**2) + np.divide(-97./1200-logR/20,R**3) + np.divide(71./3675+logR/35,R**4) + np.divide(-1./3136-logR/56,R**5) + np.divide(-1271./211680+logR/84,R**6)
  else:
    return .5*(-1+R*(-9-7*R+np.pi**2*(1+R)**2)-R*(1+R)**2*logR+np.divide((1+R)*np.log(1+R)*(1+R*(-3+(-5+R)*R)+3*R**2*(1+R)*np.log(1+R)),R)+6*R*(1+R)**2*spence(1+R))
@njit
def velocity_dispersion_squared(R):
  return 3*velocity_dispersion_radial_squared(R)
@njit
def KE_circular(R):
    if R < 0.1:
      return R/4.-R**2/3.+3.*R**3/8-2.*R**4/5+5.*R**5/12-3.*R**6/7
    else:
      return np.divide(mass(R),2*R)
@njit
def d2density_dpotential2(R):
  return np.divide(R**3*(R*(-2+4*R-R**3+R**4)-2*(-1+R+2*R**2)*np.log(1+R)),(1+R)**2*(-R+(1+R)*np.log(1+R))**3)
@njit
def dPsidR(R):
  return (R/(1 + R) - np.log(1 + R))/R**2
@njit
def d2PsidR2(R):
  return (-((R*(2 + 3*R))/(1 + R)**2) + 2*np.log(1 + R))/R**3
@njit
def F(E): # fitting form from Widrow 2000
  l = 5./2
  F0 = 9.1968e-2
  q = -2.7419
  p = np.array([.362,-.5639,-.0859,-.4912])
  P = 0.
  for i,pi in enumerate(p):
    P += pi*E**(i+1)
  if E <= 0 or E >= 1:
    return 0
  val = F0*np.power(E,1.5)*np.power(1-E,-l)
  if E > 0.99:
    val *= (1+(1-E)/2.+(1-E)**2/3.+(1-E)**3/4.+(1-E)**4/5.)**q
  else:
    val *= (np.divide(-np.log(E),1-E))**q
  return val*np.exp(P)
@njit
def F_aniso(E,L,Ra):
  Q = E-L**2/(2*Ra**2)
  l = 5./2
  if Ra == 0.6:
    F0 = 1.0885e-1
    q = -1.0468
    p = np.array([-1.6805,18.360,-151.72,336.71,-288.09,85.472])
  elif Ra == 1:
    F0 = 3.8287e-2
    q = -1.0389
    p = np.array([0.3497,-12.253,-9.1225,101.15,-127.43,47.401])
  elif Ra == 3:
    F0 = 4.2486e-3
    q = -1.0385
    p = np.array([0.7577,-25.283,149.27,-282.53,229.13,-69.048])
  elif Ra == 10:
    F0 = 3.8951e-4
    q = -1.0447
    p = np.array([-2.2679,79.474,-237.74,329.07,-223.43,59.581])
  P = 0
  for i,pi in enumerate(p):
    P += pi*Q**(i+1)
  if Q <= 0 or Q >= 1:
    return 0
  val = F0*np.power(Q,-.5)*np.power(1-Q,-l)
  if Q > 0.99:
    val *= (1+(1-Q)/2.+(1-Q)**2/3.+(1-Q)**3/4.+(1-Q)**4/5.)**q
  else:
    val *= (np.divide(-np.log(Q),1-Q))**q
  return val*np.exp(P)
@njit
def F_reduced(E): # fitting form from Widrow 2000
  F0 = 9.1968e-2
  q = -2.7419
  p = np.array([.362,-.5639,-.0859,-.4912])
  P = 0
  for i,pi in enumerate(p):
    P += pi*E**(i+1)
  if E <= 0 or E >= 1:
    return 0
  val = F0*np.power(E,1.5)
  if E > 0.99:
    val *= (1+(1-E)/2.+(1-E)**2/3.+(1-E)**3/4.+(1-E)**4/5.)**q
  else:
    val *= (np.divide(-np.log(E),1-E))**q 
  return val*np.exp(P)