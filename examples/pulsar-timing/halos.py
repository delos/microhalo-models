import numpy as np
import sys

sys.path.insert(0, '../..')
import halo_formation

N = 1000
k,pk = np.loadtxt('power_32_20.txt').T

c = halo_formation.Cosmology(k,pk,Rmin=.03,Rmax=100,nr=300,method='ta')

A,r,M,ac = c.sample(N,return_ace=True)
idx = (M > 0)&(A > 0)

# rho_s, r_s from r_max, M_max assuming NFW profile
rhos = M[idx]/(0.58097*r[idx]**3)
rs = r[idx]/2.16258

# peak number density -> halo number density
n = c.n * np.sum(idx)/idx.size

np.savetxt('halos_32_20.txt',np.stack((rhos,rs)).T,fmt='%.3e',
           header='rho_s (Msun/kpc^3), r_s (kpc) [n=%.4e kpc^-3]'%n)
