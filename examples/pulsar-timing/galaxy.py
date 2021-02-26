import numpy as np
from numba import njit
from scipy.interpolate import interp1d

import table_bulge

R0 = 8. # kpc
V0 = 226. # km/s
V_LSR = V0

# stellar density functions ---------------------------------------------------

# THIN DISK

@njit
def diskfun_old(R,z,rho0,eps):
  a2 = R**2 + (z/eps)**2
  hRp2 = 25. # kpc^2
  hRm2 = 9. # kpc^2
  d0 = 0.07648875260822759 # = (np.exp(-a2/hRp2) - np.exp(-a2/hRm2)) for a=8
  return rho0/d0 * (np.exp(-a2/hRp2) - np.exp(-a2/hRm2))

@njit
def diskfun_young(R,z,rho0,eps):
  a2 = R**2 + (z/eps)**2
  hRp2 = 6.4009 # kpc^2
  hRm2 = 1.7424 # kpc^2
  d0 = 0.03842207507582092 # = (np.exp(-np.sqrt(0.25+a2/hRp2)) - np.exp(-np.sqrt(0.25+a2/hRm2))) for a=8
  return rho0/d0 * (np.exp(-np.sqrt(0.25+a2/hRp2)) - np.exp(-np.sqrt(0.25+a2/hRm2)))

rho0_disk = np.array([4.0e6,7.9e6,6.2e6,4.0e6,5.8e6,4.9e6,6.6e6,3.96e6])
eps_disk = np.array([0.0140,0.0268,0.0375,0.0551,0.0696,0.0785,0.0791])
flare_disk = np.array([9.5,5.4e-4])
@njit
def disk(R,z,i):
  #if R > trunc_disk:
  #  return 0.
  if R > flare_disk[0]:
    z /= 1 + flare_disk[1] * (R-flare_disk[0])
  if i == 0:
    return diskfun_old(R,z,rho0_disk[0],eps_disk[0])
  elif i < 7:
    return diskfun_young(R,z,rho0_disk[i],eps_disk[i])
  elif i == 7:
    dens = diskfun_old(R,z,rho0_disk[0],eps_disk[0])
    dens0 = rho0_disk[0]
    for j in range(1,7):
      dens += diskfun_young(R,z,rho0_disk[j],eps_disk[j])
      dens0 += rho0_disk[j]
    return dens * rho0_disk[7]/dens0 # distribute WD

@njit
def disk_vec(R,z,i):
  #if R > trunc_disk:
  #  return 0.
  idx_flare = np.where(R>flare_disk[0])
  z[idx_flare] /= 1. + flare_disk[1] * (R[idx_flare]-flare_disk[0])
  if i == 0:
    return diskfun_old(R,z,rho0_disk[0],eps_disk[0])
  elif i < 7:
    return diskfun_young(R,z,rho0_disk[i],eps_disk[i])
  elif i == 7:
    dens = diskfun_old(R,z,rho0_disk[0],eps_disk[0])
    dens0 = rho0_disk[0]
    for j in range(1,7):
      dens += diskfun_young(R,z,rho0_disk[j],eps_disk[j])
      dens0 += rho0_disk[j]
    return dens * rho0_disk[7]/dens0 # distribute WD

# THICK DISK

rho0_thickdisk = np.array([1.34e6,3.04e5])
hR_thickdisk = 2.5 # kpc
hz_thickdisk = 0.8 # kpc
xl_thickdisk = 0.4 # kpc
zfac_thickdisk = (1/hz_thickdisk)/(xl_thickdisk*(2.+xl_thickdisk/hz_thickdisk))
zfac2_thickdisk = np.exp(xl_thickdisk/hz_thickdisk)/(1+xl_thickdisk/(2*hz_thickdisk))

@njit
def thickdisk(R,z,i):
  ret = rho0_thickdisk[i] * np.exp(-(R-R0)/hR_thickdisk)
  if np.abs(z) <= xl_thickdisk:
    return ret * (1-zfac_thickdisk*z**2)
  return ret * zfac2_thickdisk * np.exp(-np.abs(z)/hz_thickdisk)

@njit
def thickdisk_vec(R,z,i):
  ret = rho0_thickdisk[i] * np.exp(-(R-R0)/hR_thickdisk)
  for j,z_ in enumerate(z):
    if np.abs(z_) <= xl_thickdisk:
      ret[j] *= (1-zfac_thickdisk*z_**2)
    else:
      ret[j] *= zfac2_thickdisk * np.exp(-np.abs(z_)/hz_thickdisk)
  return ret

# SPHEROID

ac2_spheroid = 0.25 # kpc^2
rho0_spheroid = 9.32e3
eps_spheroid = 0.76

@njit
def spheroid(R,z):
  a2 = R**2 + (z/eps_spheroid)**2
  if a2 <= ac2_spheroid:
    return rho0_spheroid * (ac2_spheroid/R0**2)**-1.22
  else:
    return rho0_spheroid * (a2/R0**2)**-1.22

@njit
def spheroid_vec(R,z):
  a2 = R**2 + (z/eps_spheroid)**2
  a2[np.where(a2<ac2_spheroid)] = ac2_spheroid
  return rho0_spheroid * (a2/R0**2)**-1.22

# BULGE

def bulge(R,z):
  return table_bulge.rho(R,z)
def bulge_vec(R,z):
  return table_bulge.rho_vec(R,z)

# stellar velocities ----------------------------------------------------------

sigma_disk = np.array([
  [16.7,19.8,27.2,30.2,36.7,43.1,43.1],
  [10.8,12.8,17.6,19.5,23.7,27.8,27.8],
  [ 6.0, 8.0,10.0,13.2,15.8,17.4,17.5],
  ]).T
sigma_thickdisk = np.array([67.,51.,42.])
sigma_spheroid = np.array([131.,106.,85.])
sigma_bulge = np.array([113.,115.,100.])

# dlnsigma^2/dR = sigmagrad
# ln sigma^2 = sigmagrad*R + C
# sigma^2 = exp(sigmagrad*R + C)
# sigma = sigma0 * exp(0.5*sigmagrad*(R-R0))
sigmagrad_disk = -0.1

#sigma_disk = np.sqrt(np.sum(sigma_disk**2,axis=1))
sigma_thickdisk = np.sqrt(np.sum(sigma_thickdisk**2))
sigma_spheroid = np.sqrt(np.sum(sigma_spheroid**2))
sigma_bulge = np.sqrt(np.sum(sigma_bulge**2))

r_sigma_disk_ = np.geomspace(1e-2,1e3,300)
sigma_disk_ = []
interp_sigma_disk = []
for j in range(7):
  sigma_disk_ += [sigma_disk[j].reshape((3,-1)) * np.array([np.exp(0.5*sigmagrad_disk*(r_sigma_disk_-R0)),np.full_like(r_sigma_disk_,1.),np.full_like(r_sigma_disk_,1.)])]
  sigma_disk_[j] = np.sqrt(np.sum(sigma_disk_[j]**2,axis=0))
  interp_sigma_disk += [interp1d(r_sigma_disk_,sigma_disk_[j],bounds_error=False,fill_value=(sigma_disk_[j][0],sigma_disk_[j][-1]))]
  

ad_disk = np.array([3.5,3.1,5.8,7.3,10.8,14.8,14.8])
ad_thickdisk = 53.
ad_spheroid = 226.
ad_bulge = 79.

@njit
def sample_velocity(Vrel,sigma,Vmax):
  N = Vrel.size
  Vrel_ = np.zeros(N)
  for i in range(N):
    while True:
      vx = np.random.normal(0,sigma)
      vy = np.random.normal(0,sigma)
      vz = np.random.normal(0,sigma)
      Vrel_[i] = np.sqrt((Vrel[i]+vx)**2+vy**2+vz**2)
      Pv = Vrel_[i]/Vmax
      if Pv > np.random.rand():
        break
  return Vrel_
@njit
def sample_velocity_sigmavec(Vrel,sigma,Vmax):
  N = Vrel.size
  Vrel_ = np.zeros(N)
  for i in range(N):
    while True:
      vx = np.random.normal(0,sigma[i])
      vy = np.random.normal(0,sigma[i])
      vz = np.random.normal(0,sigma[i])
      Vrel_[i] = np.sqrt((Vrel[i]+vx)**2+vy**2+vz**2)
      Pv = Vrel_[i]/Vmax
      if Pv > np.random.rand():
        break
  return Vrel_