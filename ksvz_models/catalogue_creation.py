# create_catalogues.py

import h5py as h5
import numpy as np
import os.path
import time

from itertools import combinations_with_replacement
from numba import njit
from sympy.utilities.iterables import multiset_partitions
from tqdm import trange

from .model_building import encalc_times_36, find_LP, repinfo

@njit
def compute_eon_values(models: np.ndarray[int], repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
   eonvals = []
   for reps in models:
      e, n = encalc_times_36(reps, repinfo)
      eonvals.append([e, n])
   return np.array(eonvals, dtype='int')

def save_initial_catalogue(masses: list[float], reps: list[int]) -> None:
   for q in [1, 2]:
      t0 = time.time()
      models = np.array(list(combinations_with_replacement(reps, q)), dtype='int')
      print(f"Created combinations for N_Q = {q:d}...", flush=True)
      eonvals = compute_eon_values(models, repinfo)
      for i,mQ in enumerate(masses):
         h5name = f"output/data/addNQ{q:d}_m{i:d}.h5"
         lps = []
         for model in models:
               lp, _ = find_LP(model, mQ, plot=False)
               lps.append(lp)
         with h5.File(h5name, 'w') as f:
               f.attrs['LP_threshold'] = np.inf
               f.attrs['m_Q'] = mQ
               f.create_dataset("model", data=models, dtype='i2')
               f.create_dataset("E", data=eonvals[:,0], dtype='i4')
               f.create_dataset("N", data=eonvals[:,1], dtype='i4')
               f.create_dataset("LP", data=lps, dtype='f8')
      print("Computed {:d} models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(len(models), q, len(masses), (time.time()-t0)/60), flush=True)

@njit
def extend_additive_models(models: np.ndarray[int]) -> np.ndarray[int]:
   """
   Extend the models by one new heavy quark.

   Parameters
   ----------
   models : np.ndarray[int]
      An (m,n)-array of m models with n heavy quarks to extend

   Returns
   -------
   np.ndarray[int]
      An (m,n+1)-array of extended models with n+1 heavy quarks

   Notes
   -----
   - To avoid double couting, we only add representations with the lables up the label of the value of the first one
   """
   new_models = [[i]+list(mod) for mod in models for i in range(1, mod[0]+1)]
   return np.array(new_models, dtype='int')

def create_extended_catalogue(nq_max: int, lp_threshold: float = 1.0e18, verbose: bool = True) -> None:
   if nq_max < 3:
      raise ValueError("nq_max must be at least 3.")
   if not os.path.isfile("output/data/addNQ2_m0.h5"):
      raise FileNotFoundError("Initial catalogues not found for N_Q = 2.")
   t0 = time.time()
   if verbose:
      qrange = trange(3, nq_max+1)
   else:
      qrange = range(3, nq_max+1)
   for q in qrange:
      t1 = time.time()
      existing_cats = [f for f in os.listdir("output/data/") if f.startswith(f"addNQ{(q-1):d}_m")]
      n_masses = len(existing_cats)
      n_models = 0
      for i in range(n_masses):
         h5name_old = f"output/data/addNQ{(q-1):d}_m{i:d}.h5"
         h5name_new = f"output/data/addNQ{q:d}_m{i:d}.h5"
         with h5.File(h5name_old, 'r') as f:
            mQ = f.attrs['m_Q']
            lp_threshold_old = f.attrs['LP_threshold']
            if lp_threshold_old < lp_threshold:
               raise ValueError(f"LP threshold for {h5name_old} is lower (more restrictive) than the new threshold.")
            models = f["model"][:]
            lps = f["LP"][:]
            cond = (lps >= lp_threshold)
            models_to_extend = models[cond]
         if len(models_to_extend) > 0:
            new_models = extend_additive_models(models_to_extend)
            n_models += len(new_models)
            eonvals = compute_eon_values(new_models, repinfo)
            lps = []
            for model in new_models:
               lp, _ = find_LP(model, mQ, plot=False, verbose=verbose)
               lps.append(lp)
            with h5.File(h5name_new, 'w') as f:
               f.attrs['LP_threshold'] = lp_threshold
               f.attrs['m_Q'] = mQ
               f.create_dataset("model", data=new_models, dtype='i2')
               f.create_dataset("E", data=eonvals[:,0], dtype='i4')
               f.create_dataset("N", data=eonvals[:,1], dtype='i4')
               f.create_dataset("LP", data=lps, dtype='f8')
      print("Computed {:d} models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(n_models, q, n_masses, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)

@njit
def extend_all_models(set1: list[int], set2: list[int], dat: np.ndarray[int], repinfo: np.ndarray[int] = repinfo) -> list[int]:
   if len(set1) >= len(set2):
      reps = list(set1) + [-r for r in set2]
   else:
      reps = list(set2) + [-r for r in set1]
   e, n = encalc_times_36(reps, repinfo)
   rew_row = np.array([e, n] + reps, dtype='int')
   return rew_row

def create_full_catalogue(nq_max: int, lp_threshold: float = 1.0e18, verbose: bool = True) -> None:
   if nq_max < 2:
      raise ValueError("nq_max must be at least 2.")
   t0 = time.time()
   if verbose:
      qrange = trange(2, nq_max+1)
   else:
      qrange = range(2, nq_max+1)
   for q in qrange:
      t1 = time.time()
      existing_cats = [f for f in os.listdir("output/data/") if f.startswith(f"addNQ{q:d}_m")]
      n_masses = len(existing_cats)
      n_models = 0
      for i in range(n_masses):
         h5name_old = f"output/data/addNQ{q:d}_m{i:d}.h5"
         h5name_new = f"output/data/allNQ{q:d}_m{i:d}.h5"
         with h5.File(h5name_old, 'r') as f:
            mQ = f.attrs['m_Q']
            lp_threshold_old = f.attrs['LP_threshold']
            if lp_threshold_old < lp_threshold:
               raise ValueError(f"LP threshold for {h5name_old} is lower (more restrictive) than the new threshold.")
            cond = (f["LP"][:] >= lp_threshold)
            evals = f["E"][cond]
            nvals = f["N"][cond]
            models_to_extend = f["model"][cond]
            if verbose:
               print(f"Reading file {h5name_old} with {len(models_to_extend):d} models to extend", flush=True)
         data = np.column_stack((evals,nvals,models_to_extend))
         if len(models_to_extend) > 0:
            n_models -= len(data)
            for mod in models_to_extend:
               # N.B. The multiset_partitions generator expects a list as the first argument
               for set1,set2 in multiset_partitions(list(mod),2):
                  new_row = extend_all_models(set1, set2, data, repinfo)
                  data = np.vstack((data, new_row))
            n_models += len(data)
            with h5.File(h5name_new, 'w') as f:
               f.attrs['LP_threshold'] = lp_threshold
               f.attrs['m_Q'] = mQ
               f.create_dataset("model", data=data[:,2:], dtype='i2')
               f.create_dataset("E", data=data[:,0], dtype='i4')
               f.create_dataset("N", data=data[:,1], dtype='i4')
         elif verbose:
            print(f"No models to extend for N_Q = {q:d} and mass {mQ:.2e} GeV.", flush=True)
      print("Computed {:d} models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(n_models, q, n_masses, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)