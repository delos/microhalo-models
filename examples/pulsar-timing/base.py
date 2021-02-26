import numpy as np
import sys
sys.path.insert(0, '../tools/')
import sample_halo
import pk_EMDE

def run(argv):
  N = 100000
  TRH = 400 # MeV
  Rcut = 20

  if len(argv) < 4:
    print('python script.py <N> <TRH/MeV> <Rcut>')
    return 0
  N = int(argv[1])
  TRH = float(argv[2])
  Rcut = float(argv[3])

  filename = 'base/halos_%g_%g.txt'%(TRH,Rcut)

  k,pk = pk_EMDE.power(TRH*1e-3,Rcut) # k in Mpc^-1
  k *= 1e-3 # Mpc^-1 -> kpc^-1

  c = sample_halo.Cosmology(k,pk,Rmin=.03,Rmax=Rcut*3,nr=300,method='ta')

  A,r,M,ac = c.sample(N,return_ace=True)

  rhos = M/(0.58097*r**3)
  rs = r/2.16258
  
  idx = (M > 0)&(A > 0)
  n = c.n*r[idx].size/r.size
  
  np.savetxt(filename,np.stack((rhos[idx],rs[idx])).T,fmt='%.3e',header='rho_s (Msun/kpc^3), r_s (kpc) [T_RH=%g MeV, x_cut=%g, n=%.4e kpc^-3]'%(TRH,xcut,n))

if __name__ == '__main__':
  #from sys import argv
  from mpi4py import MPI

  # initializing MPI
  comm = MPI.COMM_WORLD
  n_proc = comm.Get_size()
  proc_id = comm.Get_rank()

  N = 200000
  #TRHs = [3.2,5.6]#[32,320,18,180,56,560]
  #xcuts = [5,10,15,20,25,30,35,40]
  TRHs = [3.2,5.6,32,320,18,180,56,560]
  TRHs = [10,100,1000]
  xcuts = [5,10]#[15,20,25,30,35,40]

  if proc_id < len(TRHs)*len(xcuts):
    TRH = TRHs[proc_id//len(xcuts)]
    xcut = xcuts[proc_id%len(xcuts)]
    argv = [0,N,TRH,xcut]
    run(argv)
