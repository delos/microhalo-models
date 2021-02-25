import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import quad,cumtrapz,trapz,solve_ivp
from scipy.optimize import fmin,brentq,fixed_point
import scipy.linalg as linalg
import time

# helper function for the peak number density (BBKS 1986)
def f(x):
  return np.where(x>.03,
                  .5*(x**3-3*x)*(erf(2.5**.5*x)+erf(2.5**.5*.5*x))+(2./(5*np.pi))**.5*((7.75*x**2+1.6)*np.exp(-.625*x**2)+(.5*x**2-1.6)*np.exp(-2.5*x**2)),
                  3**5*5**1.5/(7*2**11*(2*np.pi)**.5)*x**8*(1-.625*x**2)
                  )

# moments of the power spectrum
def sigmaj2(j,k,Pk):
  integrand = Pk*k**(2*j)
  return trapz(integrand,x=np.log(k),axis=0)
def sigmaj(j,k,Pk):
  return np.sqrt(sigmaj2(j,k,Pk))

# ellipsoidal collapse threshold
def ec_func(f,e,p):
  return 1+0.47*(5*(e**2-p*np.abs(p))*f**2)**0.615
def ec_scale(e,p):
  func = lambda f: ec_func(f,e,p)
  try:
    return fixed_point(func,1.1,maxiter=25)
  except:
    return 0

dc = 3./5*(3*np.pi/2)**(2./3)        # linear delta at collapse
dv = 3./5*(3./4 + 9*np.pi/8)**(2./3) # linear delta at virialization
dt = 3./5*(3*np.pi/4)**(2./3)        # linear delta at turnaround

# window functions
def sinc(x):
  return np.where(x > 0.1, np.divide(np.sin(x),x,where=x>0.1), 1. - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.)
def W(x):
  return 3*np.where(x > 0.1, np.divide(np.sin(x)-x*np.cos(x),x**3,where=x>0.1), 1./3. - x**2/30. + x**4/840. - x**6/45360. + x**8/3991680.)

# peak ellipticity/prolateness distributions
def fep(e,p,nu):
  return 1125./np.sqrt(10*np.pi)*e*(e**2-p**2)*nu**5*np.exp(-5./2*nu**2*(3*e**2+p**2))
def fe(e,nu):
  return 45*e*np.exp(-10*e**2*nu**2)*nu**2*(e*np.sqrt(10./np.pi)*nu+np.exp(5./2*e**2*nu**2)*(-1+5*e**2*nu**2)*erf(np.sqrt(5./2)*e*nu))
def Fe(e,nu):
  return -3*np.exp(-10*e**2*nu**2)*e*np.sqrt(10/np.pi)*nu+np.exp(-15./2*e**2*nu**2)*(1-15*e**2*nu**2)*erf(np.sqrt(5./2)*e*nu)+erf(np.sqrt(10)*e*nu)
def Fp(e,p,nu):
  return (np.exp(-5./2*p**2*nu**2)*(10*(e*np.exp(5./2*p**2*nu**2)+np.exp(5./2*e**2*nu**2)*p)*nu+np.exp(5./2*(e**2+p**2)*nu**2)*np.sqrt(10*np.pi)*(-1+5*e**2*nu**2)*(erf(np.sqrt(5./2)*e*nu)+erf(np.sqrt(5./2)*p*nu))))/(2*(10*e*nu+np.exp(5./2*e**2*nu**2)*np.sqrt(10*np.pi)*(-1+5*e**2*nu**2)*erf(np.sqrt(5./2)*e*nu)))
  
class Cosmology(object):
  
  '''
  Class for the generation of halo populations using the method in
  arXiv:1905.05766. Instantiate the class once per cosmological scenario and
  then use it to generate halos.
  
  Parameters:
    
    k, pk: a tabulation of the dimensionless matter power spectrum P(k), scaled
    such that P(k,a) = P(k) a^{2g} during matter domination. Here, g ~ 0.9 due
    to the noncontribution of baryons to clustering at microhalo scales (g=1
    if baryons are neglected).
    
    a: the scale factor at which the halo population is desired. Can be
    changed later. Default 1.
    
    method: the method for determining r_max and M_max (radius of and mass
    enclosing maximum circular velocity). See Section IV of arXiv:1905.05766
    for detail. Possible values are 'turnaround', 's=0', and 's=1' currently.
    Can be changed later. Default 's=1'.
    
    numax: only sample peaks with height delta < numax*sigma, where sigma is
    the rms variance of the density field. Must be finite. Default 7.
    
    nr: resolution of radius grid to use in sampling r_max, M_max. Default
    nr=300 radius points.
    
    Rmin: minimum radius of grid, in units of the characteristic comoving
    correlation length Rstar (typically, Rstar is roughly the reciprocal of the
    power spectrum cutoff wavenumber). Default 0.03.
    
    Rmax: maximum radius of grid, in units of Rstar. Material that initially
    lies beyond the comoving radius Rmax*Rstar is neglected in sampling r_max
    and M_max. Default 300.
    
    OmegaM: present-day matter density in units of the critical density.
    Default 0.3089.
    
    OmegaB: present-day baryon density in units of the critical density.
    Default 0.048859.
    
    rhoCrit: present-day critical density. Default 127.313454.
    
    no_cov: if True, do not generate covariance matrices. Covariance matrices
    are only required to sample r_max and M_max. Default False.
  
  Methods:
    
    sample_A(N): sample N instances of the r^-3/2 asymptotic coefficients A.
    
    sample_rM(N): sample N instances of r_max and M_max.
    
    sample(N): jointly sample A with r_max and M_max.
    
    sample_profile():
      
    sample_peak(N):
      
    set_scale(a): change the scale factor at which to generate halos.
    
    set_method(method): change the method used to sample r_max and M_max.
  '''
  
  # distributions
  def dndnudx(self,nu,x):
    return np.exp(-.5*nu**2)/((2*np.pi)**2*self.Rstar**3)*f(x)*np.exp(-.5*(x-self.gamma*nu)**2/(1.-self.gamma**2))/np.sqrt(2*np.pi*(1-self.gamma**2))
  def dndx(self,x):
    return f(x)/(2*(2*np.pi)**2*self.Rstar**3)*np.exp(-x**2/2.)*(1+erf(x*self.gamma/(2*(1-self.gamma**2))**.5))
  
  def _covariance(self):  
    self.r = np.geomspace(self.Rmin,self.Rmax,self.nr)*self.Rstar
    self.rm = self.r[1:]
    self.lnr = np.log(self.r)
    self.lnrm = np.log(self.rm)
    
    # give everything a leading r index (nr)
    pshape = (self.nk, 1, 1,) # k index, for integration
    vshape = ( 1,self.nr, 1,) # vector index
    kr = self.k.reshape(pshape)*self.r.reshape(vshape) # vector index
    skr = sinc(kr)
    
    time_start = time.clock()
    # column vectors, shape (1,nr,1)
    self.cov_delta_nu = 1./self.sigma0*trapz(self.pk.reshape(pshape)       *skr,x=self.lnk,axis=0)
    self.cov_delta_x  = 1./self.sigma2*trapz((self.pk*self.k**2).reshape(pshape)*skr,x=self.lnk,axis=0)
    # matrices, shape (1,nr,nr)
    self.cov_delta_delta = trapz(self.pk.reshape(pshape)*skr*skr.reshape((self.nk,1,self.nr)),x=self.lnk,axis=0)
    del kr, skr
    # delta(r) distribution
    delta_cov = self.cov_delta_delta-1./(1-self.gamma**2)*(np.matmul(self.cov_delta_nu,self.cov_delta_nu.T)+np.matmul(self.cov_delta_x,self.cov_delta_x.T)-self.gamma*(np.matmul(self.cov_delta_nu,self.cov_delta_x.T)+np.matmul(self.cov_delta_x,self.cov_delta_nu.T)))
    self.delta_vals, self.delta_vecs = linalg.eigh(delta_cov) # vecs.T@vecs = 1, vecs.T@M@vecs = np.diag(vals), vecs@np.diag(vals)@vecs.T = M
    self.delta_vals[self.delta_vals<0]=0
    print('  covariance matrices computed in %.2fs'%(time.clock() - time_start))
    
  def set_scale(self,a):
    
    '''
    Change the scale factor at which to generate the halo population.
    '''
    
    self.a = a
    self.dcoll = dc/self.a**self.g
    self.dvir = dc/self.a**self.g
    
  def set_method(self,method):
    
    '''
    Change the method used to sample r_max and M_max.
    '''
    
    if method in ['ta' or 'turnaround']:
      self.s = 'ta'
      self.beta = .131
      self.betaM = 0.273
    elif method in ['s=0']:
      self.s = 0
      self.beta = .414
      self.betaM = 0.441
    elif method in ['s=1']:
      self.s = 1
      self.beta = .846
      self.betaM = 0.658
    else:
      raise ValueError('method must be turnaround, s=0, or s=1')
  
  def __init__(self,k,pk,a=1,method='s=1',numax=7,nr=300,Rmin=0.03,Rmax=300,
               OmegaM=0.3089,OmegaB=0.048859,rhoCrit=127.313454,no_cov=False,):
    
    # power spectrum table
    self.k = k
    self.pk = pk
    self.nk = len(k)
    self.lnk = np.log(k)
    
    # power spectrum properties
    self.sigma0 = sigmaj(0,k,pk)
    self.sigma1 = sigmaj(1,k,pk)
    self.sigma2 = sigmaj(2,k,pk)
    self.gamma = self.sigma1**2/(self.sigma0*self.sigma2)
    self.Rstar = 3**.5*self.sigma1/self.sigma2
    
    # cosmology
    self.g = 5./4*(1-24./25*OmegaB/OmegaM)**.5-1./4
    self.rhoC = rhoCrit*(OmegaM-OmegaB)
    
    # total number density, reduced distributions
    self.n = quad(self.dndx,0,np.inf)[0]
    self.fx = lambda x: self.dndx(x)/self.n
    self.fnux = lambda nu,x: self.dndnudx(nu,x)/self.n
    
    # halo generation options
    self.set_scale(a)
    self.set_method(method)
    self.alpha = 12.1
    self.numax = numax
    self.nr = nr
    self.Rmin = Rmin
    self.Rmax = Rmax
    
    # prepare covariance matrices
    if not no_cov:
      self._covariance()
    
  def init(self,*args,**kwargs):
    self.__init__(*args,**kwargs)
  
  def sample_nux(self,N):
    
    '''
    Sample peak properties nu and x (height and "steepness") as defined in BBKS
    1986.
    '''
    
    # use rejection method
    fxmax = self.fx(fmin(lambda x: -self.fx(x),1,disp=False)[0])
    fnuxmax = self.fnux(*fmin(lambda nux: -self.fnux(nux[0],nux[1]),[1,1],disp=False))
    def randnux(N):
      P = np.random.rand(N)*fxmax
      x = np.random.rand(N)*self.numax
      x = x[P<self.fx(x)]
      N = len(x)
      P = np.random.rand(N)*fnuxmax
      nu = np.random.rand(N)*self.numax
      idx = P<self.fnux(nu,x)
      return nu[idx],x[idx]
    Ngen = 0
    nu = []
    x = []
    time_start = time.clock()
    while(Ngen < N):
      nu_,x_ = randnux(N)
      nu = np.concatenate((nu,nu_))
      x = np.concatenate((x,x_))
      Ngen = len(nu)
    print('  nu, x sampled in%.2fs'%(time.clock() - time_start))
    return nu[:N], x[:N]
  
  def sample_ep(self,nu,x):
    
    '''
    Sample peak properties e and p (ellipticity and prolateness) as defined in
    BBKS 1986.
    '''
    
    # use inverse transform method
    N = len(nu)
    P = np.random.rand(N,2)
    e = np.zeros(N)
    p = np.zeros(N)
    time_start = time.clock()
    for i in range(N):
      e[i] = brentq(lambda e: Fe(e,nu[i])-P[i,0],0,2./nu[i])#,rtol=1e-3)
      p[i] = brentq(lambda p: Fp(e[i],p,nu[i])-P[i,1],-e[i],e[i])#,rtol=1e-3)
    print('  e, p sampled in %.2fs'%(time.clock() - time_start))
    return e,p
    
  def _sample_A(self,nu,x,e,p,return_ac=False,return_ace=False):
    N = len(nu)
    
    # compute asymptote
    d = nu*self.sigma0
    d2d = x*self.sigma2
    #A = d**2.25*d2d**-.75*self.rhoC
    A = dc**(1.5*(1-1./self.g))*d**(.75*(2./self.g+1))*d2d**-.75*self.rhoC
    ec_mod = np.zeros(N)
    time_start = time.clock()
    for i in range(N):
      ec_mod[i] = ec_scale(e[i],p[i])
    print('  A sampled in %.2fs'%(time.clock() - time_start))
    idx = (ec_mod>0)&(d>self.dcoll*ec_mod)
    Ae = A.copy()
    Ae[~idx] = -1
    Ae[idx] *= ec_mod[idx]**(-1.5/self.g)*self.alpha
    if not (return_ac or return_ace):
      return Ae
    else:
      ret = [Ae]
      if return_ac:
        ret += [(dc/d)**(1./self.g)]
      if return_ace:
        ret += [(ec_mod*dc/d)**(1./self.g)]
      return ret
  
  def _sample_delta(self,nu,x,return_Delta=False,return_eps=False):
    delta_mean = 1./(1-self.gamma**2)*((self.cov_delta_nu-self.gamma*self.cov_delta_x)*nu+(self.cov_delta_x-self.gamma*self.cov_delta_nu)*x)
    kappa = np.random.normal(0,self.delta_vals**.5).reshape((self.nr,1))
    delta = (delta_mean+np.matmul(self.delta_vecs,kappa))
    delta.shape = (self.nr)
    if not (return_Delta or return_eps):
      return delta
    else:
      d = nu*self.sigma0
      Delta = 3./self.rm**3*(cumtrapz(delta*self.r**2,x=self.r)+d*self.r[0]**3/3.)
      ret = [delta]
      if return_Delta:
        ret += [Delta]
      if return_eps:
        ret += [1.-delta[1:]/Delta]
      return tuple(ret)
  
  def _profile(self,d,delta,Delta,eps,return_X=False,return_dlnXdlnq=False):
    fail = [-1,-1]
    if return_X:
      fail += [-1]
    if return_dlnXdlnq:
      fail += [-1]
    if d < self.dcoll:
      return fail
    # interpolation
    d_interp = interp1d(self.lnr,delta,kind='cubic',fill_value='extrapolate')
    D_interp = interp1d(self.lnrm,Delta,kind='cubic',fill_value='extrapolate')
    e_interp = lambda lnr: 1-d_interp(lnr)/D_interp(lnr)
    # get limits
    try:
      ic1 = np.where(Delta<self.dvir)[0][0]
      lnrc1 = brentq(lambda lnr: D_interp(lnr)-self.dvir,self.lnrm[ic1-1],self.lnrm[ic1])
    except:
      ic1 = len(self.rm)-1
      lnrc1 = self.lnrm[-1]
    try:
      ic2 = np.where((eps<0)&(self.rm>self.Rstar))[0][0]
      lnrc2 = brentq(lambda lnr: e_interp(lnr),self.lnrm[ic2-1],self.lnrm[ic2])
    except:
      ic2 = len(self.rm)-1
      lnrc2 = self.lnrm[-1]
    ic = min(ic1,ic2)
    lnrc = min(lnrc1,lnrc2)
    # adiabatic contraction
    def fun_dXdlnq(lnq,X):
      return -X*(3-(3*(3-self.s)*e_interp(lnq)/self.g-self.s)*(X-1))/(1+(4-self.s)*(X-1))
    try:
      sol = solve_ivp(fun_dXdlnq,[lnrc,self.lnrm[0]],[1],
                      t_eval=self.lnrm[:ic][::-1],
                      #method='Radau',jac=jac_dQdlnrL,
                      )
      X = sol.y[0][::-1]
    except Exception as e:
      print(e)
      return fail
    if not sol.success:
      print(sol.message)
      return fail
    M0 = 4./3*np.pi*self.rm[:ic]**3*self.rhoC
    rf0 = self.rm[:ic]/Delta[:ic]**(1./self.g)
    M = M0*X*self.betaM
    rf = rf0/X*self.beta
    ret = [rf,M]
    if return_X:
      ret += [X]
    if return_dlnXdlnq:
      ret += [fun_dXdlnq(self.lnrm[:ic],X)/X]
    return tuple(ret)
  
  def _profile_simple(self,d,delta,Delta,eps):
    fail = [-1,-1]
    if d < self.dcoll:
      return fail
    # get limits
    try:
      ic1 = np.where(Delta<self.dvir)[0][0]
    except:
      ic1 = len(self.rm)-1
    try:
      #ic2 = np.where((np.diff(Delta)>0)&(self.rm[:-1]>self.Rstar))[0][0]
      ic2 = np.where((eps<0)&(self.rm>self.Rstar))[0][0]-1
    except:
      ic2 = len(self.rm)-1
    ic = min(ic1,ic2)
    M = self.betaM*4./3*np.pi*self.rm[:ic]**3*self.rhoC
    rf = self.beta*self.rm[:ic]/Delta[:ic]**(1./self.g)
    ret = [rf,M]
    return tuple(ret)
  
  def _rM(self,eps,rf,M,dlnXdlnq=0):
    ith = np.argmax(M/rf)
    if ith == len(rf)-1:
      # max is at the largest resolved radius---this is not the true r_max.
      return 0,0
    dlnMdlnr = (3.+dlnXdlnq)/(1+3*eps/self.g-dlnXdlnq)
    if dlnMdlnr[ith] < 1:
      i2 = ith
      i1 = i2-1
    else:
      i1 = ith
      i2 = i1+1
    rmax = (rf[i1]*(dlnMdlnr[i2]-1)+rf[i2]*(1-dlnMdlnr[i1]))/(dlnMdlnr[i2]-dlnMdlnr[i1])
    Mmax = (M[i1]*(dlnMdlnr[i2]-1)+M[i2]*(1-dlnMdlnr[i1]))/(dlnMdlnr[i2]-dlnMdlnr[i1])
    return rmax,Mmax
  
  def _sample_rM(self,nu,x):
    d = nu*self.sigma0
    rmax = np.zeros_like(nu)-1.
    Mmax = np.zeros_like(nu)-1.
    time_start = time.clock()
    for i in range(len(nu)):
      if i%1000==0:
        print('  sampled %d r_max, M_max pairs in %.2fs'%(i,time.clock() - time_start))
      if d[i] < self.dcoll:
        continue
      delta,Delta,eps = self._sample_delta(nu[i],x[i],True,True)
      if self.s == 'ta':
        rf,M = self._profile_simple(d[i],delta,Delta,eps)
        dlnXdlnq = 0
      else:
        rf,M,dlnXdlnq = self._profile(d[i],delta,Delta,eps,return_dlnXdlnq=True)
      if np.size(rf) > 1:
        rmax[i],Mmax[i] = self._rM(eps[:len(rf)],rf,M,dlnXdlnq)
    return rmax, Mmax
  
  def sample_A(self,N,return_ac=False,return_ace=False):
    
    '''
    Sample the r^-3/2 asymptotic coefficients A.
    
    The unit of A is the unit of rhoCrit*k^(-3/2).
    
    Parameters:
      
      N: number of peaks to sample.
      
      return_ac: also return the spherical collapse scale factor for the set
      of peaks. Default False.
      
      return_ace: also return the ellipsoidal collapse scale factor for the
      set of peaks. Default False.
      
    Returns:
      
      A: the array of N asymptotic coefficients A associated with the peaks
      sampled. A=-1 if the peak does not collapse by the scale factor a.
      
      [if return_ac=True] ac: the spherical collapse scale factors associated
      with the peaks sampled.
      
      [if return_ace=True] ace: the ellipsoidal collapse scale factors
      associated with the peaks sampled.
    '''
    
    nu,x = self.sample_nux(N)
    e,p = self.sample_ep(nu,x)
    A = self._sample_A(nu,x,e,p,return_ac=return_ac,return_ace=return_ace)
    if return_ac and return_ace:
      return A[0],A[1],A[2]
    elif return_ac or return_ace:
      return A[0],A[1]
    return A
  
  def sample_rM(self,N):
    
    '''
    Sample r_max and M_max.
    
    The unit of r_max is the unit of k^-1, while the unit of M_max is the unit
    of rhoCrit*k^-3.
    
    Parameters:
      
      N: number of peaks to sample.
      
    Returns:
      
      r_max, M_max: the radius of maximum circular velocity and the mass
      enclosed therein. r_max=M_max=-1 if the peak does not collapse by
      the scale factor a. r_max=M_max=0 if r_max is too large to be resolved
      with the provided input parameter Rmax.
    '''
    
    nu,x = self.sample_nux(N)
    rmax, Mmax = self._sample_rM(nu,x)
    return rmax,Mmax
    
  def sample(self,N,return_ac=False,return_ace=False):
    
    '''
    Jointly sample A with r_max and M_max. Note that these quantities are
    correlated.
  
    
    The unit of A is the unit of rhoCrit*k^(-3/2), the unit of r_max is the
    unit of k^-1, and the unit of M_max is the unit of rhoCrit*k^-3.
    
    Parameters:
      
      N: number of peaks to sample.
      
      return_ac: also return the spherical collapse scale factor for the set
      of peaks. Default False.
      
      return_ace: also return the ellipsoidal collapse scale factor for the
      set of peaks. Default False.
      
    Returns:
      
      A: the array of N asymptotic coefficients A associated with the peaks
      sampled. A=-1 if the peak does not collapse by the scale factor a.
      
      r_max, M_max: the radius of maximum circular velocity and the mass
      enclosed therein. r_max=M_max=-1 if the peak does not collapse by
      the scale factor a. r_max=M_max=0 if r_max is too large to be resolved
      with the provided input parameter Rmax. Note that for technical reasons
      it is possible for r_max=M_max=-1 while A!=-1.
      
      [if return_ac=True] ac: the spherical collapse scale factors associated
      with the peaks sampled.
      
      [if return_ace=True] ace: the ellipsoidal collapse scale factors
      associated with the peaks sampled.
    '''
    
    nu,x = self.sample_nux(N)
    e,p = self.sample_ep(nu,x)
    A = self._sample_A(nu,x,e,p,return_ac=return_ac,return_ace=return_ace)
    rmax, Mmax = self._sample_rM(nu,x)
    if return_ac and return_ace:
      return A[0],rmax,Mmax,A[1],A[2]
    elif return_ac or return_ace:
      return A[0],rmax,Mmax,A[1]
    return A,rmax,Mmax
  
  def sample_profile(self,return_d=False,return_D=False,return_A=False):
    
    '''
    Generate a single randomly sampled halo (enclosed) mass profile M(r). This
    profile is only calibrated to simulations at r_max and M_max and does not
    capture the inner r^(-3/2) asymptote.
    
    The unit of r is the unit of k^-1, while the unit of M is the unit of
    rhoCrit*k^-3.
    
    Parameters:
      
      return_d: also return the density profile of the initial peak.
      
      return_D: also return the enclosed mass profile of the initial peak.
      
      return_A: also return the asymptotic coefficient A associated with this
      peak.
      
    Returns:
      
      r, M: the halo's mass profile.
      
      [if return_d=True] d: the fractional density contrast about the initial
      peak, as a function of comoving radius. The corresponding radii are in
      the array r belonging to this class.
      
      [if return_D=True] D: the fractional enclosed mass contrast about the
      initial peak, as a function of comoving radius. The corresponding radii
      are in the array r belonging to this class.
      
      [if return_A=True] A: the asymptotic coefficient A associated with the
      peak sampled. A=-1 if the peak does not collapse by the scale factor a.
    '''
    
    nu,x = self.sample_nux(1)
    d = nu*self.sigma0
    delta,Delta,eps = self._sample_delta(nu,x,True,True)
    if self.s == 'ta':
      profile = self._profile_simple(d,delta,Delta,eps)
    else:
      profile = self._profile(d,delta,Delta,eps)
    if return_d:
      profile = tuple(list(profile)+[delta])
    if return_D:
      profile = tuple(list(profile)+[Delta])
    if return_A:
      e,p = self.sample_ep(nu,x)
      A = self._sample_A(nu,x,e,p)
      profile = tuple(list(profile)+[A])
    return profile
  
  def sample_peak(self,N):
    
    '''
    Sample peak properties nu (height), x ("steepness"), e (ellipticity), and
    p (prolateness) as defined in BBKS 1986.
    
    Parameters:
      
      N: number of peaks to sample.
      
    Returns:
      
      nu, x, e, p.
    '''
    
    nu,x = self.sample_nux(N)
    e,p = self.sample_ep(nu,x)
    return nu,x,e,p
