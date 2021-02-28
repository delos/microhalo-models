import numpy as np

'''
Dimensionful -> dimensionless parameters
'''

def x_func(R,rhos,Rs,Rhos): # |Eb|/[(F/R)r^2]
  r = R/Rs
  F = Rhos*Rs * (np.log(1+r)-r/(1+r))/r**2 # F / (4piG)
  Eb = rhos*np.log(2) # Eb / (4piG r_s^2)
  DE = F/R # DE / (4piG r_s^2)
  return Eb/DE

# orbital timescale
def t0_func(R,Rs,Rhos,G):
  r = R/Rs
  F = Rhos*Rs * (np.log(1+r)-r/(1+r))/r**2 # F / (4piG)
  return np.sqrt(R/(4*np.pi*G*F))