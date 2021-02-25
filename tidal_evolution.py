import tidal.orbit_functions_nfw as functions
import tidal.reduced_orbit_definitions as defs
import tidal.tidal_evolution_helpers as helpers

def J_evolve_compress(R,Rc,eta,Rs,Rhos,rhos,age,G=4.3022682e-6,decay_cap=None):
  
  '''
  Evolve a subhalo's J factor (proportional to annihilation rate). This
  function includes both period-averaged time evolution (described in
  arXiv:1906.10690) and radius-based periodic evolution (described in Appendix
  C of arXiv:1910.08553).
  
  This function is unit-agnostic as long as R, Rc, and Rs have the same units
  and Rhos, rhos, and age^-2*G^-1 have the same units.
  
  Parameters:
    
    R: subhalo's instantaneous orbital radius about the host.
    
    Rc: subhalo's circular orbit radius,the radius of the circular orbit
    with the same orbital energy.
    
    eta: subhalo's orbital circularity. eta = L/Lc, where L is the orbital
    angular momentum and Lc is the angular momentum of the circular orbit with
    the same energy.
    
    Rs: host's NFW scale radius.
    
    Rhos: host's NFW scale density.
    
    rhos: subhalo's initial NFW scale density.
    
    age: duration of subhalo's tidal evolution.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    decay_cap: if not None, the maximum allowed value of -d log J / d log t.
    The value decay_cap=1 has some physical motivation (see arXiv:1906.10690),
    but the impact of this parameter is marginal.
    
  Returns:
    
    The factor by which the subhalo's J factor is scaled due to tidal
    evolution.
  '''
  
  rc = Rc/Rs
  T = defs.t0_func(Rc,Rs,Rhos,G)*functions.Tfun(rc,eta)
  z = functions.z2_fun(rc,eta,rhos/Rhos,1,1)
  x = functions.xE_fun(rc,eta,defs.x_func(Rc,rhos,Rs,Rhos))
  y = functions.ravg_fun(rc,eta)
  Ravg = y*Rs
  return helpers.scale_J_time(age/T,x,y,z,decay_cap)*helpers.scale_J_radius(R/Ravg,x,y)

def J_evolve(Rc,eta,Rs,Rhos,rhos,age,G=4.3022682e-6,decay_cap=None):
  
  '''
  Evolve a subhalo's J factor (proportional to annihilation rate). This
  function includes only (period-averaged) time evolution as described in
  arXiv:1906.10690.
  
  This function is unit-agnostic as long as Rc and Rs have the same units
  and Rhos, rhos, and age^-2*G^-1 have the same units.
  
  Parameters:
    
    Rc: subhalo's circular orbit radius,the radius of the circular orbit
    with the same orbital energy.
    
    eta: subhalo's orbital circularity. eta = L/Lc, where L is the orbital
    angular momentum and Lc is the angular momentum of the circular orbit with
    the same energy.
    
    Rs: host's NFW scale radius.
    
    Rhos: host's NFW scale density.
    
    rhos: subhalo's initial NFW scale density.
    
    age: duration of subhalo's tidal evolution.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    decay_cap: if not None, the maximum allowed value of -d log J / d log t.
    The value decay_cap=1 has some physical motivation (see arXiv:1906.10690),
    but the impact of this parameter is marginal.
    
  Returns:
    
    The factor by which the subhalo's J factor is scaled due to tidal
    evolution.
  '''
  
  rc = Rc/Rs
  T = defs.t0_func(Rc,Rs,Rhos,G)*functions.Tfun(rc,eta)
  z = functions.z2_fun(rc,eta,rhos/Rhos,1,1)
  x = functions.xE_fun(rc,eta,defs.x_func(Rc,rhos,Rs,Rhos))
  y = functions.ravg_fun(rc,eta)
  return helpers.scale_J_time(age/T,x,y,z,decay_cap)

def J_compress(R,Rc,eta,Rs,Rhos,rhos):
  
  '''
  Evolve a subhalo's J factor (proportional to annihilation rate). This
  function includes solely radius-based evolution (tidal compression) as
  described in Appendix C of arXiv:1910.08553.
  
  This function is unit-agnostic as long as R, Rc, and Rs have the same units
  and Rhos and rhos have the same units.
  
  Parameters:
    
    R: subhalo's instantaneous orbital radius about the host.
    
    Rc: subhalo's circular orbit radius,the radius of the circular orbit
    with the same orbital energy.
    
    eta: subhalo's orbital circularity. eta = L/Lc, where L is the orbital
    angular momentum and Lc is the angular momentum of the circular orbit with
    the same energy.
    
    Rs: host's NFW scale radius.
    
    Rhos: host's NFW scale density.
    
    rhos: subhalo's initial NFW scale density.
    
  Returns:
    
    The factor by which the subhalo's J factor is scaled due to tidal
    compression.
  '''
  
  rc = Rc/Rs
  x = functions.xE_fun(rc,eta,defs.x_func(Rc,rhos,Rs,Rhos))
  y = functions.ravg_fun(rc,eta)
  Ravg = y*Rs
  return helpers.scale_J_radius(R/Ravg,x,y)

def rmax_evolve(Rc,eta,Rs,Rhos,rhos,age,G=4.3022682e-6,decay_cap=None):
  
  '''
  Evolve a subhalo's rmax, the radius of maximum circular velocity (within
  the subhalo), according to the model described in Appendix E of
  arXiv:1906.10690.
  
  This function is unit-agnostic as long as Rc and Rs have the same units
  and Rhos, rhos, and age^-2*G^-1 have the same units.
  
  Parameters:
    
    Rc: subhalo's circular orbit radius,the radius of the circular orbit
    with the same orbital energy.
    
    eta: subhalo's orbital circularity. eta = L/Lc, where L is the orbital
    angular momentum and Lc is the angular momentum of the circular orbit with
    the same energy.
    
    Rs: host's NFW scale radius.
    
    Rhos: host's NFW scale density.
    
    rhos: subhalo's initial NFW scale density.
    
    age: duration of subhalo's tidal evolution.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    decay_cap: if not None, the maximum allowed value of -d log rmax / d log t.
    
  Returns:
    
    The factor by which the subhalo's rmax is scaled due to tidal evolution.
  '''
  
  rc = Rc/Rs
  T = defs.t0_func(Rc,Rs,Rhos,G)*functions.Tfun(rc,eta)
  z = functions.z2_fun(rc,eta,rhos/Rhos,1,1)
  x = functions.xE_fun(rc,eta,defs.x_func(Rc,rhos,Rs,Rhos))
  y = functions.ravg_fun(rc,eta)
  return helpers.scale_r_time(age/T,x,y,z,decay_cap)

def vmax_evolve(Rc,eta,Rs,Rhos,rhos,age,G=4.3022682e-6,decay_cap=None):
  
  '''
  Evolve a subhalo's vmax, the maximum circular velocity (within the subhalo),
  according to the model described in Appendix E of arXiv:1906.10690.
  
  This function is unit-agnostic as long as Rc and Rs have the same units
  and Rhos, rhos, and age^-2*G^-1 have the same units.
  
  Parameters:
    
    Rc: subhalo's circular orbit radius,the radius of the circular orbit
    with the same orbital energy.
    
    eta: subhalo's orbital circularity. eta = L/Lc, where L is the orbital
    angular momentum and Lc is the angular momentum of the circular orbit with
    the same energy.
    
    Rs: host's NFW scale radius.
    
    Rhos: host's NFW scale density.
    
    rhos: subhalo's initial NFW scale density.
    
    age: duration of subhalo's tidal evolution.
    
    G: gravitational constant; default value is in (km/s)^2 kpc/Msun.
    
    decay_cap: if not None, the maximum allowed value of -d log vmax / d log t.
    
  Returns:
    
    The factor by which the subhalo's vmax is scaled due to tidal evolution.
  '''
  
  rc = Rc/Rs
  T = defs.t0_func(Rc,Rs,Rhos,G)*functions.Tfun(rc,eta)
  z = functions.z2_fun(rc,eta,rhos/Rhos,1,1)
  x = functions.xE_fun(rc,eta,defs.x_func(Rc,rhos,Rs,Rhos))
  y = functions.ravg_fun(rc,eta)
  return helpers.scale_v_time(age/T,x,y,z,decay_cap)