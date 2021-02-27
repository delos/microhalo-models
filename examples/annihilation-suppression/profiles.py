import numpy as np
from scipy.special import spence

log2 = np.log(2)
log4 = np.log(4)

supported_params = [
    [1,3,1], # NFW
    [1,3,1.5], # Moore
    [2,3,0], # cored
]

def density_norm(params):
  if params == [1,3,1]:
    return 1.
  elif params == [1,3,1.5]:
    return .5
  elif params == [2,3,0]:
    return 1.
def density(R,params):
  if params == [1,3,1]:
    return 1./(R*(1+R)**2)
  elif params == [1,3,1.5]:
    return .5/(R**1.5*(1+R)**1.5)
  elif params == [2,3,0]:
    return 1./(1+R**2)**1.5
def mass(R,params):
  if params == [1,3,1]:
    return np.where(R<.1,
                    R**2/2. - 2.*R**3/3. + 3.*R**4/4 - 4*R**5/5 + 5.*R**6/6,
                    np.log(1+R)-R/(1.+R)
                    )
  elif params == [1,3,1.5]:
    return np.where(R<.1,
                    R**1.5/3. - 3.*R**2.5/10 + 15.*R**3.5/56 - 35*R**4.5/144 + 315.*R**5.5/1408,
                    np.arcsinh(R**.5)-(R/(1.+R))**.5
                    )
  elif params == [2,3,0]:
    return np.where(R<.1,
                    R**3/3. - 3.*R**5/10 + 15.*R**7/56,
                    np.arcsinh(R)-R/(1.+R**2)**.5
                    )
def integrated_density_over_r(R,params):
  if params == [1,3,1]:
    return R/(1.+R)
  elif params == [1,3,1.5]:
    return np.sqrt(R/(1.+R))
  elif params == [2,3,0]:
    return 1.-1./np.sqrt(1.+R**2)
def density_mean(R,params):
  if params == [2,3,0]:
    return np.where(R<.1,
                    1.-9.*R**2/10+45.*R**4/56-35.*R**6/48+945.*R**8/1408,
                    np.divide(mass(R,params)*3.,R**3,where=R>0)
                    )
  else:
    return mass(R,params)*3./R**3
def r3_over_mass(R,params):
  if params == [1,3,1]:
    return np.where(R<.1,
                    2*R + 8*R**2/3. + 5*R**3/9. - 8*R**4/135. + 17*R**5/810. - 86*R**6/8505.,
                    R**3/mass(R,params)
                    )
  elif params == [1,3,1.5]:
    return np.where(R<.1,
                    3*R**1.5 + 27*R**2.5/10 + 27*R**3.5/1400 + 493*R**4.5/14000 - 1231029*R**5.5/43120000,
                    R**3/mass(R,params)
                    )
  elif params == [2,3,0]:
    return np.where(R<.1,
                    3 + 27*R**2/10. + 27*R**4/1400. + 493*R**6/14000. - 1231029*R**8/43120000.,
                    R**3/mass(R,params)
                    )
def potential(R,params):
  if params == [1,3,1]:
    return np.where(R<.1,
                    1.-R/2.+R**2/3.-R**3/4.+R**4/5.-R**5/6.+R**6/7.,
                    np.divide(np.log(1+R),R,where=R>0)
                    )
  elif params == [1,3,1.5]:
    return np.where(R<.1,
                    1.-2.*R**.5/3.+R**1.5/5.-3.*R**2.5/28.+5.*R**3.5/72.-35.*R**4.5/704.+63.*R**5.5/1664.,
                    np.divide(R-(R+R**2)**.5+np.arcsinh(R**.5),R,where=R>0)
                    )
  elif params == [2,3,0]:
    return np.where(R<.1,
                    1.-R**2/6.+3.*R**4/40.-5.*R**6/112.+35.*R**8/1152.,
                    np.divide(np.arcsinh(R),R,where=R>0)
                    )
def velocity_dispersion_radial_squared(R,params):
  logR = np.log(R,where=R>0)
  if params == [1,3,1]:
    return np.where(R<.1,
                    .25*(-23+2*np.pi**2-2*logR)*R+(-59./6+np.pi**2-logR)*R**2+1./24*(-101+12*np.pi**2-12*logR)*R**3+(11*R**4)/60.-(13*R**5)/240.+(37*R**6)/1400.,
                    np.where(R>10.,
                             np.divide(-3./16+logR/4,R,where=R>10.) + np.divide(69./200+logR/10,R**2,where=R>10.) + np.divide(-97./1200-logR/20,R**3,where=R>10.) + np.divide(71./3675+logR/35,R**4,where=R>10.) + np.divide(-1./3136-logR/56,R**5,where=R>10.) + np.divide(-1271./211680+logR/84,R**6,where=R>10.),
                             .5*(-1+R*(-9-7*R+np.pi**2*(1+R)**2)-R*(1+R)**2*logR+np.divide((1+R)*np.log(1+R)*(1+R*(-3+(-5+R)*R)+3*R**2*(1+R)*np.log(1+R)),R,where=R>0)+6*R*(1+R)**2*spence(1+R))
                             )
                    )
  elif params == [1,3,1.5]:
    return np.where(R<.1,
                    R**.5/3+1./150*(817-960*log2+120*logR)*R**1.5+(8683./1400-48*log2/5+6*logR/5)*R**2.5+(19861./25200-12*log2/5+3*logR/10)*R**3.5+(-461803./2217600+2*log2/5-logR/20)*R**4.5+(1576591./19219200-3*log2/20+3*logR/160)*R**5.5,
                    np.where(R>10.,
                             np.divide(-7+4*(log4+logR),32*R,where=R>10.) + 3.*np.divide(49+20*(log4+logR),1600*R**2,where=R>10.) + np.divide(-79-420*(log4+logR),19200*R**3,where=R>10.) + np.divide(-3589+3780*(log4+logR),268800*R**4,where=R>10.) + np.divide(48311-27720*(log4+logR),2867200*R**5,where=R>10.) + np.divide(-285041+120120*(log4+logR),17203200*R**6,where=R>10.),
                             .2*(R*(1+R))**1.5*(5./(1+R)+np.divide(-2.+7*R,R**2,where=R>0)+np.divide(2*(1+2*R*(-1+4*R+8*R**2))*np.arcsinh(R**.5),np.sqrt(R**5*(1+R)),where=R>0)+4*np.log(R/(256*(1+R)**5),where=R>0))
                             )
                    )
  elif params == [2,3,0]:
    return np.where(R<.1,
                    1.5-2*log2+(25./12-3*log2)*R**2+(41./80-3*log2/4)*R**4+(-269./3360+log2/8)*R**6+(2171./80640-3*log2/64)*R**8,
                    np.where(R>10.,
                             np.divide(-3-12*log2+8*log4+4*logR,16*R,where=R>10.) + np.divide(5+12*log2+12*logR,96*R**3,where=R>10.) + np.divide(13-24*log2-24*logR,512*R**5,where=R>10.) + np.divide(-391+360*log2+360*logR,15360*R**7,where=R>10.),
                             np.divide((2+6*R**2+4*R**4)*np.arcsinh(R)+R*(1+R**2)**.5*(1-2*(1+R**2)*np.log(4*(1+R**2))),2*R,where=R>0)
                             )
                    )
  
def velocity_dispersion_squared(R,params):
  return 3*velocity_dispersion_radial_squared(R,params)

def KE_circular(R,params):
  if params == [1,3,1]:
    return np.where(R<.1,
                    R/4.-R**2/3.+3.*R**3/8-2.*R**4/5+5.*R**5/12-3.*R**6/7,
                    np.divide(mass(R,params),2*R,where=R>0)
                    )
  elif params == [1,3,1.5]:
    return np.where(R<.1,
                    R**.5/6.-3.*R**1.5/20+15.*R**2.5/112-35.*R**3.5/288+315.*R**4.5/2816-693.*R**5.5/6656,
                    np.divide(mass(R,params),2*R,where=R>0)
                    )
  elif params == [2,3,0]:
    return np.where(R<.1,
                    R**2/6.-3.*R**4/20+15.*R**6/112-35.*R**8/288,
                    np.divide(mass(R,params),2*R,where=R>0)
                    )

def d2density_dpotential2(R,params):
  if params == [1,3,1]:
    return np.divide(R**3*(R*(-2+4*R-R**3+R**4)-2*(-1+R+2*R**2)*np.log(1+R)),(1+R)**2*(-R+(1+R)*np.log(1+R))**3)
  elif params == [1,3,1.5]:
    return np.divide(R**3*(3-12*R+R**3-2*R**4)+3*R**2.5*np.sqrt(1+R)*(-1+4*R)*np.arcsinh(np.sqrt(R)),8*(1+R)**4*(np.sqrt(R/(1+R))-np.arcsinh(np.sqrt(R)))**3)
  elif params == [2,3,0]:
    return np.divide(R**4*(R*(-6+9*R**2-2*R**4+R**6)-3*np.sqrt(1+R**2)*(-2+3*R**2)*np.arcsinh(R)),(1+R**2)**2.5*(-R+np.sqrt(1+R**2)*np.arcsinh(R))**3)

def dPsidR(R,params):
  if params == [1,3,1]:
    return (R/(1 + R) - np.log(1 + R))/R**2
  elif params == [1,3,1.5]:
    return (np.sqrt(R/(1 + R)) - np.arcsinh(np.sqrt(R)))/(R**2)
  elif params == [2,3,0]:
    return (R/np.sqrt(1 + R**2) - np.arcsinh(R))/R**2

def d2PsidR2(R,params):
  if params == [1,3,1]:
    return (-((R*(2 + 3*R))/(1 + R)**2) + 2*np.log(1 + R))/R**3
  elif params == [1,3,1.5]:
    return -((4 + 5*R)/(2*R**2.5*(1 + R)**1.5)) + (2*np.arcsinh(np.sqrt(R)))/R**3
  elif params == [2,3,0]:
    return (-2*R - 3*R**3 + 2*(1 + R**2)**1.5*np.arcsinh(R))/(R**3*(1 + R**2)**1.5)

def F(E,params): # fitting form from Widrow 2000
  if params == [1,3,1]:
    l = 5./2
    F0 = 9.1968e-2
    q = -2.7419
    p = np.array([.362,-.5639,-.0859,-.4912])
  elif params == [1,3,1.5]:
    l = 9./2
    F0 = 4.8598e-1
    q = -2.8216
    p = np.array([.3526,-5.199,3.5461,-.884])
  elif params == [2,3,0]:
    l = 0.
    F0 = 5.8807e-2
    q = -2.6312
    p = np.array([-3.7147,41.045,-132.2,216.9,-170.23,51.606])
  P = 0
  for i,pi in enumerate(p):
    P += pi*E**(i+1)
  return (
      F0*np.power(E,1.5,where=E>0)*np.power(1-E,-l,where=E<1)
      *np.where(E>0.99,1+(1-E)/2.+(1-E)**2/3.+(1-E)**3/4.+(1-E)**4/5.,np.divide(-np.log(E,where=E>0),1-E,where=E<1))**q
      *np.exp(P)
      )

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
  return (
      F0*np.power(Q,-.5,where=Q>0)*np.power(1-Q,-l,where=Q<1)
      *np.where(Q>0.99,1+(1-Q)/2.+(1-Q)**2/3.+(1-Q)**3/4.+(1-Q)**4/5.,np.divide(-np.log(Q,where=Q>0),1-Q,where=Q<1))**q
      *np.exp(P)
      )

def F_reduced(E,params): # fitting form from Widrow 2000
  if params == [1,3,1]:
    F0 = 9.1968e-2
    q = -2.7419
    p = np.array([.362,-.5639,-.0859,-.4912])
  elif params == [1,3,1.5]:
    F0 = 4.8598e-1
    q = -2.8216
    p = np.array([.3526,-5.199,3.5461,-.884])
  elif params == [2,3,0]:
    F0 = 5.8807e-2
    q = -2.6312
    p = np.array([-3.7147,41.045,-132.2,216.9,-170.23,51.606])
  P = 0
  for i,pi in enumerate(p):
    P += pi*E**(i+1)
  return (
      F0*np.power(E,1.5,where=E>0)
      *np.where(E>0.99,1+(1-E)/2.+(1-E)**2/3.+(1-E)**3/4.+(1-E)**4/5.,np.divide(-np.log(E,where=E>0),1-E,where=E<1))**q
      *np.exp(P)
      )

def F_lambda(params): # fitting form from Widrow 2000
  if params == [1,3,1]:
    return 5./2
  elif params == [1,3,1.5]:
    return 9./2
  elif params == [2,3,0]:
    return 0.
