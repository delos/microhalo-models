import numpy as np
from scipy.special import hyp2f1

def from_nfw(rhos_NFW,rs_NFW):
  
  '''
  Convert from NFW profile to postencounter density profile. See
  arXiv:1907.13133 for the profile and arXiv:xxxx.xxxxx for a refinement.
  
  Parameters:
  
    rhos_NFW, rs_NFW: scale parameters associated with the initial NFW profile.
    
  Returns:
    
    rhos, rs: scale parameters associated with the initial postencounter
    density profile.
  '''
  
  return rhos_NFW*1.17, rs_NFW*.86

def stellar_encounters_norelax(rhos,rs,m_list,b_list,v_list,G=4.3022682e-6,
                               Nmax=100000,):
  
  '''
  Apply stellar encounters with no dynamical relaxation between them. This
  function is much faster to evaulate than stellar_encounters().
  
  Simulations in Appendix D of arXiv:1910.08553 suggest that it is appropriate
  to neglect relaxation for halos that are also subjected to tidal evolution.
  
  This function is unit-agnostic as long as the mass and length units of rhos,
  rs, m_list, b_list, and v_list^2/G are all consistent.
  
  Parameters:
  
    rhos, rs: scale parameters associated with the initial postencounter
    density profile (related to the NFW profile via the from_nfw function
    above).
    
    m_list, b_list, v_list: masses, impact parameters, and relative velocities
    associated with the stellar encounters.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    Nmax: Maximum number of encounters to explicitly treat (to save time).
    
  Returns:
      
    rhos, rs: scale parameters associated with the final postencounter density
    profile 
  '''
  
  qsum = 0
  N = b_list.size
  if N > Nmax:
    indices = np.sort(np.random.choice(np.arange(N),Nmax,replace=False))
    fac = N * 1./Nmax
  else:
    indices = np.arange(N)
    fac = 1.
  for j in indices:
    b = b_list[j]
    m = m_list[j]
    V = v_list[j]
    q = G/(2*np.pi)*m**2/(rhos*V**2)*1./(b**4+rs**4) * fac

    dr = ((1+((q+qsum)/.35)**.63) / (1+((qsum)/.35)**.63))**-1.5873

    rs *= dr
    rhos *= dr**-.72

    qsum += q
  return rhos,rs

def stellar_encounters(rhos,rs,m_list,b_list,v_list,t_list,G=4.3022682e-6,
                       relaxation_coefficient=2.,return_tracks=False):
  
  '''
  Apply stellar encounters based on the model in arXiv:1907.13133.
  
  This function is unit-agnostic as long as the mass and length units of rhos,
  rs, m_list, b_list, v_list^2/G, and t_list^2*G are all consistent.
  
  Parameters:
  
    rhos, rs: scale parameters associated with the initial postencounter
    density profile (related to the NFW profile via the from_nfw function
    above).
    
    m_list, b_list, v_list, t_list: masses, impact parameters, relative
    velocities, and times associated with the stellar encounters.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    relaxation_coefficient: the relaxation coefficient "lambda" defined in
    arXiv:1907.13133. Essentially, this is the number of dynamical times
    required for a halo to relax between encounters. The default value of 2
    matches simulation results.
    
    return_tracks: if true, return the full evolutionary history of the
    density profile; otherwise, just return the final profile.
    
  Returns:
      
    rhos, rs: scale parameters associated with the final postencounter density
    profile. If return_tracks=True, these are instead arrays of length
    len(t_list)+1 giving the scale parameters at time 0 followed by these
    parameters after the encounter at each time t in t_list.
  '''
  
  isort = np.argsort(t_list)
  m_list = m_list[isort]
  b_list = b_list[isort]
  v_list = v_list[isort]
  t_list = t_list[isort]
  
  q_list = np.zeros_like(b_list)
  rs_list = np.concatenate(([rs],np.zeros_like(b_list)))
  rhos_list = np.concatenate(([rhos],np.zeros_like(b_list)))
  
  ilast = 0
  
  for i,b in enumerate(b_list):
    t = t_list[i]
    b = b_list[i]
    m = m_list[i]
    V = v_list[i]
    q_list[i] = G/(2*np.pi)*m**2/(rhos_list[i]*V**2)*1./(b**4+rs_list[i]**4)
    
    tdyn = np.sqrt(3*np.pi/(16*G*rhos[i]))
    ilast += np.where(t_list[ilast:i+1]>t-tdyn)[0][0]
    qeff1 = np.sum(q_list[ilast:i+1])
    qeff2 = np.sum(q_list[ilast:i])
    
    dr = ((1+(qeff1/.35)**.63) / (1+(qeff2/.35)**.63))**-1.5873
    
    rs[i+1] = rs[i] * dr
    rhos[i+1] = rhos[i] * dr**-.72
  
  if return_tracks:
    return rhos,rs
  else:
    return rhos[-1], rs[-1]

def postencounter_density_profile(x):
  
  '''
  The postencounter density profile, as defined in arXiv:xxxx.xxxxx.
  
  Parameters:
    
    x = r/r_s: the radius r at which to evaluate the density, in units of the
    scale radius r_s.
    
  Returns:
    
    rho/rho_s: the density rho in units of the scale density rho_s.
  '''
  
  alpha = 0.78
  beta = 5.
  
  q = (1./3*x**alpha)**beta
  return np.exp(-1./alpha * x**alpha * (1+q)**(1-1/beta) * hyp2f1(1,1,1+1/beta,-q))/x