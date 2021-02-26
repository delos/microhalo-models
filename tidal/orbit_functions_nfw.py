import numpy as np
from scipy.optimize import brentq

# NFW profile

def mass(R):
  return np.where(R<.1,
                  R**2/2. - 2.*R**3/3. + 3.*R**4/4 - 4*R**5/5 + 5.*R**6/6,
                  np.log(1+R)-R/(1.+R)
                  )

def potential(R):
  return np.where(R<.1,
                  1.-R/2.+R**2/3.-R**3/4.+R**4/5.-R**5/6.+R**6/7.,
                  np.divide(np.log(1+R),R,where=R>0)
                  )


def r3_over_mass(R):
  return np.where(R<.1,
                  2*R + 8*R**2/3. + 5*R**3/9. - 8*R**4/135. + 17*R**5/810. - 86*R**6/8505.,
                  R**3/mass(R)
                  )

# orbit functions

def Efun(r,eta):
  p = np.array([ 3.32704302,  0.64631396,  0.88365924,  0.88088006,  0.21558338,
                3.00538329,  3.64079545,  0.08512942,  0.57028105,  0.21504841,
                1.01665749,  0.86502264,  0.50569261,  2.77393527,  0.2425718 ,
                0.64150075,  0.76628987,  0.6508455 , 18.8427342 ])
  a1,b1,c1,d1,e1,A2,a2,b2,c2,A3,a3,b3,c3,d3,e3,f3,g3,h3,i3 = p
  param1 = a1*(1+b1*np.log(1+r)-c1*r/(d1+r))/(1+e1*(np.log(1+r)-2*r/(2+r)))
  param2 = A2*(1+(r/c2)**a2)**b2
  param3 = A3*(1+(r/c3)**a3)**b3/((1+(r/f3)**d3)**e3*(1+(r/i3)**g3)**h3)
  return param1*(np.exp(param2*(1-eta**param3)))

def rfun(r,eta):
  p = np.array([0.37768543, 0.48915041, 2.41216105, 0.24256298, 0.35559159,
                1.85982781, 0.16648815])
  b,c,d,e,f,g,h = p
  return (1+b*(1-f*eta**g)*np.log(1+r)-c*r/(d*(1-h*eta)+r))/(1+e*(np.log(1+r)-2*r/(2+r)))

def Tfun(r,eta):
  A=1.730080*2
  B=0.607648
  C=0.883114
  D=2.312053
  E=0.332462
  F=0.048274
  G=1.261205
  H=0.036056
  I=1.288120
  return A*(1+F*eta**G)*(1+B*np.log(1+r)-C*r/(D+r))/(1+E*(1+H*eta**I)*(np.log(1+r)-2*r/(2+r)))

M = lambda r: 4*np.pi*mass(r)
Phi = lambda r: -4*np.pi*potential(r)
F = lambda r: M(r)/r**2
def ravg_fun(rc,eta):
  return rfun(rc,eta)*rc
def rp_fun(rc,eta):
  if np.isscalar(rc)&np.isscalar(eta):
    if eta == 0:
      return 0
    try:
      return brentq(lambda r: Phi(rc)-Phi(r)+(1-eta**2*rc**2/r**2)*M(rc)/(2*rc),1e-12,rc)
    except:
      print(rc,eta)
      raise
  else:
    ret = np.zeros(np.broadcast(rc,eta).shape)
    #rc = np.broadcast_to(rc,np.broadcast(rc,eta).shape)
    #eta = np.broadcast_to(eta,np.broadcast(rc,eta).shape)
    rc_, eta_ = np.broadcast_arrays(rc,eta)
    for i,_ in np.ndenumerate(ret):
      if eta_[i] == 0:
        ret[i] = 0
      else:
        fun = lambda r: Phi(rc_[i])-Phi(r)+(1-eta_[i]**2*rc_[i]**2/r**2)*M(rc_[i])/(2*rc_[i])
        ret[i] = brentq(fun,1e-12,rc_[i])
    return ret
def ra_fun(rc,eta):
  if np.isscalar(rc)&np.isscalar(eta):
    if eta == 0:
      return 0
    try:
      return brentq(lambda r: Phi(rc)-Phi(r)+(1-eta**2*rc**2/r**2)*M(rc)/(2*rc),rc,1e12)
    except:
      print(rc,eta)
      raise
  else:
    ret = np.zeros(np.broadcast(rc,eta).shape)
    #rc = np.broadcast_to(rc,np.broadcast(rc,eta).shape)
    #eta = np.broadcast_to(eta,np.broadcast(rc,eta).shape)
    rc_, eta_ = np.broadcast_arrays(rc,eta)
    for i,_ in np.ndenumerate(ret):
      fun = lambda r: Phi(rc_[i])-Phi(r)+(1-eta_[i]**2*rc_[i]**2/r**2)*M(rc_[i])/(2*rc_[i])
      ret[i] = brentq(fun,rc_[i],1e12)
    return ret
def z_fun(rc,eta):
  ravg = ravg_fun(rc,eta)
  rp = rp_fun(rc,eta)
  return np.where(eta>0,eta**-2*M(ravg)/M(rc)*rp**4/(ravg**3*rc),0)
def xE_fun(rc,eta,x):
  return x/Efun(rc,eta)

def tidefun(r):
  F = (np.log(1+r)-r/(1+r))/r**3 # F / (4piG)
  Fp = ((2+3*r)/(r**2*(1+r)**2) - 2*np.log(1+r)/r**3) # dF/dR / (4piG)
  return -Fp/F

def eta_from_rc_q(rc,q): # q = rp/ra
  if q == 0 or q == 1:
    return q
  rp = brentq(lambda rp: q**2*(rc*F(rc)+2*Phi(rc)-2*Phi(rp))-rc*F(rc)-2*Phi(rc)+2*Phi(rp/q),0,rc)
  eta = rp/rc*np.sqrt(1+2*(Phi(rc)-Phi(rp))/(rc*F(rc)))
  return eta

def z2_fun(rc,eta,p,a=1,b=1): # x = Eb/(F/R)
  #p = 1./np.log(2)*x*(np.log(1+rc)-rc/(1+rc))/rc**3
  # tidal radius where m/r^3 = rhot
  #y = ravg_fun(rc,eta)
  #rhot = M(y)/y**3*a
  ra = ra_fun(rc,eta)
  rhot = M(ra)/ra**3*a
  # adiabatic radius where m/r^3 = rhoa
  rp = rp_fun(rc,eta)
  rhoa = eta**2*M(rc)*rc/rp**4*b
  # subhalo density scaled by p
  if np.isscalar(rc)&np.isscalar(eta)&np.isscalar(p):
    if eta == 0:
      return 0
    rhoinv = lambda r: r3_over_mass(r)/(4*np.pi*p)
    try:
      #rho = lambda r: p*M(r)/r**3
      #rt = brentq(lambda r: rho(r)-rhot,1e-12,1e12)
      #ra = brentq(lambda r: rho(r)-rhoa,1e-12,1e12)
      rt = brentq(lambda r: rhoinv(r)-1./rhot,0,1e12)
      ra = brentq(lambda r: rhoinv(r)-1./rhoa,0,1e12)
      return ra/rt
    except:
      print('%g,%g: 1/rho(min)=%g, 1/rho(max)=%g, 1/rhot=%g, 1/rhoa=%g'%(rc,eta,rhoinv(1e-12),rhoinv(1e12),1./rhot,1./rhoa))
      raise
  else:
    ret = np.zeros(np.broadcast(rc,eta,p).shape)
    #rc = np.broadcast_to(rc,np.broadcast(rc,eta).shape)
    #eta = np.broadcast_to(eta,np.broadcast(rc,eta).shape)
    p_,rhot_,rhoa_ = np.broadcast_arrays(p,rhot,rhoa)
    for i,_ in np.ndenumerate(ret):
      #rho = lambda r: p_[i]*M(r)/r**3
      #rt = brentq(lambda r: rho(r)-rhot_[i],1e-12,1e12)
      #ra = brentq(lambda r: rho(r)-rhoa_[i],1e-12,1e12)
      rhoinv = lambda r: r3_over_mass(r)/(4*np.pi*p_[i])
      rt = brentq(lambda r: rhoinv(r)-1./rhot_[i],0,1e12)
      ra = brentq(lambda r: rhoinv(r)-1./rhoa_[i],0,1e12)
      ret[i] = ra/rt
    return ret
