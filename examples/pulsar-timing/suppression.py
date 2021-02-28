import numpy as np
import sys
from numba import njit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from orbits_nfw import randrv, Orbit
import galaxy
from imf import Mstar_mean, xicinv_vec, generate_imf

import table_fv

sys.path.insert(0, '../..')
import tidal_evolution
import stellar_encounters

G = 4.3022682e-6
Rs = 3.3875668664185703
Rhos = 397925127.43790317
#age = 13.6e9/977813106
age = 10e9/977813106

R0 = galaxy.R0
V0 = galaxy.V0
V_LSR = galaxy.V_LSR

# stellar initial mass function
a_imf = np.array([.3,1.3,2.3,2.7])
m_imf = np.array([.01,.08,.5,1])
imf = generate_imf(a_imf,m_imf)

# stellar encounter model
model_stellar = njit(stellar_encounters.stellar_encounters_norelax)

# encounters ------------------------------------------------------------------

def suppress(rs_i,rhos_i,qmin=1e-3,plot=False):
  bmax = (G/(2*np.pi)*1**2/(rhos_i*V_LSR)/qmin)**.25
  
  Mstar = Mstar_mean(*imf)
  rc,eta,mu,phi,vrel = randrv(R0,V0,1,R0,Rhos,Rs,G)
  
  i = 0
  
  o=Orbit(rc[i],eta[i],Rhos,Rs,G,R0,phi[i])
  Norb = age/o.period
  print('rc=%.3f kpc, eta=%.3f, phi=%.3f rad : %.1f periods'%(rc[i],eta[i],phi[i],Norb))
  print(  'vrel=%.3f km/s'%(vrel[i]))
  
  # apply tidal model
  
  rmax_i = 2.16258 * rs_i # coefficients for NFW profile
  vmax_i = 1.64835 * rs_i * np.sqrt(G*rhos_i)
  
  with np.errstate(divide='ignore',invalid='ignore'):
    rmax_ = rmax_i * tidal_evolution.rmax_evolve(rc[i],eta[i],Rs,Rhos,rhos_i,age)
    vmax_ = vmax_i * tidal_evolution.vmax_evolve(rc[i],eta[i],Rs,Rhos,rhos_i,age)
  
  rs_ = 0.53197 * rmax_ # coefficients for postencounter profile
  rhos_ = 1.21453 * vmax_**2 / (G * rmax_**2)
  
  # get orbit
  
  points_per_period = 1e3
  
  times = np.linspace(-age,0,int(Norb*points_per_period+.5))
  R = np.zeros_like(times)
  theta = np.zeros_like(times)
  VR = np.zeros_like(times)
  Vtheta = np.zeros_like(times)
  z = np.zeros_like(times)
  
  for j,t in enumerate(times):
    R[j],theta[j] = o.pos(t)
    VR[j],Vtheta[j] = o.vel(R[j],theta[j])
  z = o.z(R,theta)
  Vfac = 1 + (np.tan(theta) * o.cosphi)**2
  Vpara = o.cosphi/np.cos(theta) / np.sqrt(Vfac) * Vtheta
  Vperp = np.sqrt(VR**2 + o.sinphi**2/Vfac * Vtheta**2)
  Vrelfun = lambda ad: np.sqrt((Vpara-(V_LSR-ad))**2+Vperp**2)
  
  R_interp = interp1d(times,R)
  Vpara_interp = interp1d(times,Vpara)
  Vperp_interp = interp1d(times,Vperp)
  
  # differential number of encounters
  
  dNdAdt_disk = []
  Vint_disk = []
  for j in range(7):
    sigma_ = galaxy.interp_sigma_disk[j](R)
    Vint_disk += [sigma_ * table_fv.f_vec(Vrelfun(galaxy.ad_disk[j])/sigma_)]
    dNdAdt_disk += [Vint_disk[-1] * galaxy.disk_vec(R,z,j) / Mstar]
  dNdAdt_disk = np.array(dNdAdt_disk)
  dNdAdt_disk_tot = np.sum(dNdAdt_disk,axis=0)
  
  Vint_thickdisk = galaxy.sigma_thickdisk * table_fv.f_vec(Vrelfun(galaxy.ad_thickdisk)/galaxy.sigma_thickdisk)
  Vint_spheroid  = galaxy.sigma_spheroid  * table_fv.f_vec(Vrelfun(galaxy.ad_spheroid )/galaxy.sigma_spheroid )
  Vint_bulge     = galaxy.sigma_bulge     * table_fv.f_vec(Vrelfun(galaxy.ad_bulge    )/galaxy.sigma_bulge    )
  
  dNdAdt_thickdisk = Vint_thickdisk * galaxy.thickdisk_vec(R,z,0) / Mstar
  dNdAdt_spheroid =  Vint_spheroid  * galaxy.spheroid_vec(R,z)    / Mstar
  dNdAdt_bulge =     Vint_bulge     * galaxy.bulge_vec(R,z)       / Mstar
  
  dNdA_disk = cumtrapz(dNdAdt_disk,x=times,initial=0,axis=1)
  dNdA_thickdisk = cumtrapz(dNdAdt_thickdisk,x=times,initial=0)
  dNdA_spheroid = cumtrapz(dNdAdt_spheroid,x=times,initial=0)
  dNdA_bulge = cumtrapz(dNdAdt_bulge,x=times,initial=0)
  
  # number of encounters
  
  area = np.pi*bmax**2
  
  Nx_disk = dNdA_disk[:,-1] * area
  Nx_thickdisk = dNdA_thickdisk[-1] * area
  Nx_spheroid = dNdA_spheroid[-1] * area
  Nx_bulge = dNdA_bulge[-1] * area
  
  N_disk = np.zeros(7,dtype=int)
  for j in range(7):
    N_disk[j] = int(Nx_disk[j]+.5) if Nx_disk[j] > 1e4 else np.random.poisson(Nx_disk[j])
  N_thickdisk = int(Nx_thickdisk+.5) if Nx_thickdisk > 1e4 else np.random.poisson(Nx_thickdisk)
  N_spheroid = int(Nx_spheroid+.5) if Nx_spheroid > 1e4 else np.random.poisson(Nx_spheroid)
  N_bulge = int(Nx_bulge+.5) if Nx_bulge > 1e4 else np.random.poisson(Nx_bulge)
  
  print('%d encounters'%(np.sum(N_disk)+N_thickdisk+N_spheroid+N_bulge))
  
  # sample times
  
  Pt_disk = dNdA_disk / dNdA_disk[:,-2:-1]
  Pt_thickdisk = dNdA_thickdisk / dNdA_thickdisk[-1]
  Pt_spheroid = dNdA_spheroid / dNdA_spheroid[-1]

  Pt_bulge = dNdA_bulge
  if dNdA_bulge[-1] > 0:
    Pt_bulge /= dNdA_bulge[-1]
  
  St_disk = []
  for j in range(7):
    St_disk += [np.sort(interp1d(Pt_disk[j],times)(np.random.rand(N_disk[j])))]
  St_thickdisk = np.sort(interp1d(Pt_thickdisk,times)(np.random.rand(N_thickdisk)))
  St_spheroid = np.sort(interp1d(Pt_spheroid,times)(np.random.rand(N_spheroid)))
  St_bulge = np.sort(interp1d(Pt_bulge,times)(np.random.rand(N_bulge)))
  
  # get radii
  
  SR_disk = []
  for j in range(7):
    SR_disk += [R_interp(St_disk[j])]
  
  # sample velocities
  
  Sv_disk = []
  for j in range(7):
    if len(St_disk[j]) == 0:
      Sv_disk += [np.array([])]
      continue
    sigma_ = galaxy.interp_sigma_disk[j](SR_disk[j])
    vrel_ = np.sqrt((Vpara_interp(St_disk[j])-(V_LSR-galaxy.ad_disk[j]))**2+Vperp_interp(St_disk[j])**2)
    Sv_disk += [galaxy.sample_velocity_sigmavec(vrel_,sigma_,vrel_.max() + sigma_.max() * 3)]
    
  if len(St_thickdisk) > 0:
    vrel_ = np.sqrt((Vpara_interp(St_thickdisk)-(V_LSR-galaxy.ad_thickdisk))**2+Vperp_interp(St_thickdisk)**2)
    Sv_thickdisk = galaxy.sample_velocity(vrel_,galaxy.sigma_thickdisk,vrel_.max() + galaxy.sigma_thickdisk * 3)
  else:
    Sv_thickdisk = np.array([])
  
  if len(St_spheroid) > 0:
    vrel_ = np.sqrt((Vpara_interp(St_spheroid)-(V_LSR-galaxy.ad_spheroid))**2+Vperp_interp(St_spheroid)**2)
    Sv_spheroid = galaxy.sample_velocity(vrel_,galaxy.sigma_spheroid,vrel_.max() + galaxy.sigma_spheroid * 3)
  else:
    Sv_spheroid = np.array([])
  
  if len(St_bulge) > 0:
    vrel_ = np.sqrt((Vpara_interp(St_bulge)-(V_LSR-galaxy.ad_bulge))**2+Vperp_interp(St_bulge)**2)
    Sv_bulge = galaxy.sample_velocity(vrel_,galaxy.sigma_bulge,vrel_.max() + galaxy.sigma_bulge * 3)
  else:
    Sv_bulge = np.array([])
  
  # sample masses
  
  Sm_disk = []
  for j in range(7):
    Sm_disk += [xicinv_vec(np.random.rand(N_disk[j]),*imf)]
  Sm_thickdisk = xicinv_vec(np.random.rand(N_thickdisk),*imf)
  Sm_spheroid = xicinv_vec(np.random.rand(N_spheroid),*imf)
  Sm_bulge = xicinv_vec(np.random.rand(N_bulge),*imf)
  
  # sample impact parameters
  
  Sb_disk = []
  for j in range(7):
    Sb_disk += [np.random.rand(N_disk[j])**.5*bmax]
  Sb_thickdisk = np.random.rand(N_thickdisk)**.5*bmax
  Sb_spheroid = np.random.rand(N_spheroid)**.5*bmax
  Sb_bulge = np.random.rand(N_bulge)**.5*bmax
  
  # concatenate all components
  
  St = np.concatenate(St_disk+[St_thickdisk,St_spheroid,St_bulge])
  Sv = np.concatenate(Sv_disk+[Sv_thickdisk,Sv_spheroid,Sv_bulge])
  Sm = np.concatenate(Sm_disk+[Sm_thickdisk,Sm_spheroid,Sm_bulge])
  Sb = np.concatenate(Sb_disk+[Sb_thickdisk,Sb_spheroid,Sb_bulge])
  
  sort = np.argsort(St)
  St = St[sort]
  Sv = Sv[sort]
  Sm = Sm[sort]
  Sb = Sb[sort]
  
  # model
  
  rhos,rs = model_stellar(rhos_,rs_,Sm,Sb,Sv)
  
  # print stats
  
  print('scale asymptote: %g * %g = %g'%(rhos_*rs_/(rhos_i*rs_i),rhos*rs/(rhos_*rs_),rhos*rs/(rhos_i*rs_i)))
  print('scale mass: %g * %g = %g'%(rhos_*rs_**3/(rhos_i*rs_i**3),rhos*rs**3/(rhos_*rs_**3),rhos*rs**3/(rhos_i*rs_i**3)))
  print('scale density: %g * %g = %g'%(rhos_/(rhos_i),rhos/(rhos_),rhos/(rhos_i)))
  
  if plot:
    import matplotlib.pyplot as plt
    # test plot
    
    ax = plt.figure().gca()
    ax2 = plt.figure().gca()
    
    dNdA_disk_tot = np.sum(dNdA_disk,axis=0)
    ax.plot(times,dNdAdt_disk_tot,label='disk')
    ax.plot(times,dNdAdt_thickdisk,label='thick disk')
    ax.plot(times,dNdAdt_spheroid,label='halo')
    ax.plot(times,dNdAdt_bulge,label='bulge')
    ax2.plot(times,dNdA_disk_tot,label='disk')
    ax2.plot(times,dNdA_thickdisk,label='thick disk')
    ax2.plot(times,dNdA_spheroid,label='halo')
    ax2.plot(times,dNdA_bulge,label='bulge')
    
    ax.set_xlim(-age,0)
    ax2.set_xlim(-age,0)
    ax.set_xlabel(r'$t$ (s km$^{-1}$ kpc)')
    ax2.set_xlabel(r'$t$ (s km$^{-1}$ kpc)')
    ax.set_ylabel(r'$\mathrm{d}^2 N_*/(\mathrm{d} A \mathrm{d} t)$ (s$^{-1}$ km kpc$^{-3}$)')
    ax2.set_ylabel(r'$\mathrm{d} N_*/\mathrm{d} A$ (kpc$^{-2}$)')
    
    ax2.legend()
  
  orbit = {
    'rc':rc[i],
    'eta':eta[i],
    'mu':mu[i],
    'phi':phi[i],
    'vrel':vrel[i],
    }
  
  tidal = {'rs':rs_,'rhos':rhos_}
  
  stellar = {'rs':rs,'rhos':rhos}
  
  return orbit,tidal,stellar

if __name__ == '__main__':

  fname = 'halos_32_20.txt'
  
  rhos_list, rs_list = np.loadtxt(fname).T
  
  N = len(rhos_list)
  
  for i in range(N):
    print('halo %d'%i)
    orbit,tidal,stellar = suppress(rs_list[i],rhos_list[i])
    
    line = '%d  '%i
    line += '%.3e %.3e  '%(rhos_list[i],rs_list[i])
    line += '%.3e %.3e  '%(tidal['rhos'],tidal['rs'])
    line += '%.3e %.3e  '%(stellar['rhos'],stellar['rs'])
    line += '%.3e %.3f %.3f %.3f %.3e\n'%(orbit['rc'],orbit['eta'],orbit['mu'],orbit['phi'],orbit['vrel'])
    
    with open('suppressed_'+fname,'a') as fp:
      fp.write(line)
      print(line)
  print('done')

