import numpy as np
import sys
from scipy.optimize import brentq
from scipy.integrate import dblquad, cumtrapz

import halo

sys.path.insert(0, '../..')
import tidal_evolution

interp_ct = 50

rho_ratio = 1e3 # rho_s / Rho_s
dynamical_age = 50 # age * np.sqrt(G*Rho_s)
name = '%.3g_%.3g'%(rho_ratio,dynamical_age)

G = 1 #4.3022682e-6
Rs = 1
Rhos = 1

rhos = Rhos * rho_ratio
age = dynamical_age / np.sqrt(G*Rhos)

def survival_function(R,Rc,eta):
  return tidal_evolution.J_evolve_compress(R,Rc,eta,Rs,Rhos,rhos,age,G=G)

host = halo.Halo(r_s=Rs,rho_s=Rhos,G=G)

def survival_fraction(r, survival_function = lambda r,rc,eta: 1):
  Phi = host.Phi(r)
  rcmin = brentq(lambda rc: host.KEcirc(rc) + host.Phi(rc)-Phi,0,r)
  etamax = lambda rc: r/rc*np.sqrt(1.+(host.Phi(rc)-Phi)/host.KEcirc(rc))
  def df(eta,rc):
    Kc = host.KEcirc(rc)
    Pc = host.Phi(rc)
    return (4*np.pi*2**.5*host.G*host.f(Kc+Pc)
            *(host.mass_profile(rc)/(2*rc**2)+2*np.pi*rc*host.density_profile(rc))
            *eta*rc**2/r**2*Kc/np.sqrt(Kc*(1-eta**2*rc**2/r**2)+Pc-Phi)
            /host.density_profile(r) * survival_function(r,rc,eta))
  return dblquad(df,rcmin,np.inf,0,etamax,epsabs=1.49e-8,epsrel=1e-2)[0]

r_list = np.geomspace(1e-2,1e2,interp_ct)
s_list = np.zeros_like(r_list)
for j,r in enumerate(r_list):
  print('%d/%d'%(j,len(r_list)))
  try:
    s_list[j] = survival_fraction(r,survival_function)
  except Exception as e:
    print(e)

np.savetxt('tidal_scaling_%s.txt'%name,np.stack((r_list,s_list)).T,header='radius / scale radius, annihilation scaling factor')

rc_list = r_list[1:]
rho_list = 1./(r_list*(1+r_list)**2) # r**-1 at small r
m_list = np.log(1+r_list)-r_list/(1+r_list) # .5*r**2 at small r

# integrate 1/r*s[0]*(r/r[0])**.5 r**2 dr = s[0]*r[0]**-.5*r**1.5 dr
s0 = .4*s_list[0]*r_list[0]**2
msup_list = (cumtrapz(rho_list*s_list*r_list**2,x=r_list)+s0)
sc_list = msup_list/m_list[1:]

np.savetxt('tidal_scaling_cumulative_%s.txt'%name,np.stack((rc_list,sc_list)).T,header='concentration, annihilation scaling factor')