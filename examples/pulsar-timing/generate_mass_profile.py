import numpy as np
from scipy.integrate import cumtrapz
import sys
sys.path.insert(0, '../..')
import stellar_encounters

r = np.geomspace(1e-12,1e15,10000)
rho = stellar_encounters.postencounter_density_profile(r)
M = cumtrapz(4*np.pi*r**3 * rho,x=np.log(r),initial=0) + 2*np.pi*r[0]**2

np.savetxt('M.txt',np.stack((r,M)).T,header='x/r_s, M/(rho_s*r_s^3)')