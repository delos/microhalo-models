import numpy as np
import sys

sys.path.insert(0, '../..')
import halo_formation

N = 10000 # number of peaks to sample
omega = (4./3)**2 # r^-3/2 -> NFW conversion factor

k,pk = np.loadtxt('power_1GeV_40.txt').T

c = halo_formation.Cosmology(k,pk,no_cov=True)

A = c.sample_A(N)
idx = (A>0)
n = c.n*np.sum(idx)/idx.size

Ji = 4*np.pi/3*A[idx]**2
rho2 = np.mean(Ji)*n

boost = rho2/c.rhoC**2

print('annihilation is boosted by factor %.2e relative to homogeneous case'%boost)