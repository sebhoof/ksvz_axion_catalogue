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

def save_initial_catalogue(masses: list[float], reps: list[int], lp_threshold: float = 1.0e18) -> None:
   for q in [1, 2]:
      t0 = time.time()
      models = np.array(list(combinations_with_replacement(reps, q)), dtype='int')
      eonvals = compute_eon_values(models, repinfo)
      n_models = 0
      for i,mQ in enumerate(masses):
         h5name = f"output/data/addNQ{q:02d}_m{i:d}.h5"
         lps = []
         for model in models:
            lp, _ = find_LP(model, mQ, plot=False)
            lps.append(lp)
         lps = np.array(lps)
         cond = (lps >= lp_threshold)
         n_models += sum(cond)
         with h5.File(h5name, 'w') as f:
            f.attrs['LP_threshold'] = lp_threshold
            f.attrs['m_Q'] = mQ
            f.create_dataset("model", data=models[cond], dtype='i2')
            f.create_dataset("E", data=eonvals[cond,0], dtype='i4')
            f.create_dataset("N", data=eonvals[cond,1], dtype='i4')
            f.create_dataset("LP", data=lps[cond], dtype='f8')
      print("Computed {:d} valid models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(n_models, q, len(masses), (time.time()-t0)/60), flush=True)

@njit
def extend_additive_models(models: np.ndarray[int], allowed_reps: np.ndarray[int]) -> np.ndarray[int]:
   """
   Extend the models by one new heavy quark.

   Parameters
   ----------
   models : np.ndarray[int]
      An (m,n)-array of m models with n heavy quarks to extend
   allowed_reps : np.ndarray[int]
      A 1d array of allowed representations to consider

   Returns
   -------
   np.ndarray[int]
      An (m,n+1)-array of extended models with n+1 heavy quarks

   Notes
   -----
   - To avoid double couting, we only add representations with the lables up the label of the value of the first one
   """
   new_models = [[r]+list(mod) for mod in models for r in allowed_reps if r <= mod[0]]
   return np.array(new_models, dtype='int')

def create_extended_catalogue(nq_max: int, verbose: bool = True) -> None:
   if nq_max < 3:
      raise ValueError("nq_max must be at least 3.")
   if not os.path.isfile("output/data/addNQ02_m0.h5"):
      raise FileNotFoundError("Initial catalogues not found for N_Q = 2.")
   t0 = time.time()
   # Retrieve LP-allowed reps from the initial catalogue
   inital_cats = ["output/data/"+f for f in os.listdir("output/data/") if f.startswith(f"addNQ01_m")]
   allowed_reps = []
   for h5name_1 in inital_cats:
      with h5.File(h5name_1, 'r') as f:
         lp_threshold = f.attrs['LP_threshold']
         cond = (f["LP"][:] >= lp_threshold)
         allowed_reps.append(f["model"][cond].T[0])
   allowed_reps = np.array(allowed_reps, dtype='int')
   for q in range(3, nq_max+1):
      t1 = time.time()
      qold = q-1
      previous_cats = ["output/data/"+f for f in os.listdir("output/data/") if f.startswith(f"addNQ{qold:02d}_m")]
      n_masses = len(previous_cats)
      n_models = 0
      for h5name_old in previous_cats:
         i = int(h5name_old.split(f"addNQ{qold:02d}_m")[1].split(".h5")[0])
         h5name_new = f"output/data/addNQ{q:02d}_m{i:d}.h5"
         with h5.File(h5name_old, 'r') as f:
            mQ = f.attrs['m_Q']
            models = f["model"][:]
            lps = f["LP"][:]
            cond = (lps >= lp_threshold)
            models_to_extend = models[cond]
         if len(models_to_extend) > 0:
            new_models = extend_additive_models(models_to_extend, allowed_reps[i])
            eonvals = compute_eon_values(new_models, repinfo)
            lps = []
            for model in new_models:
               lp, _ = find_LP(model, mQ, plot=False, verbose=verbose)
               lps.append(lp)
            lps = np.array(lps)
            cond = (lps >= lp_threshold)
            n_models += sum(cond)
            with h5.File(h5name_new, 'w') as f:
               f.attrs['LP_threshold'] = lp_threshold
               f.attrs['m_Q'] = mQ
               f.create_dataset("model", data=new_models[cond], dtype='i2')
               f.create_dataset("E", data=eonvals[cond,0], dtype='i4')
               f.create_dataset("N", data=eonvals[cond,1], dtype='i4')
               f.create_dataset("LP", data=lps[cond], dtype='f8')
      print("Computed {:d} valid models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(n_models, q, n_masses, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)

@njit
def extend_all_models(set1: list[int], set2: list[int], repinfo: np.ndarray[int] = repinfo) -> list[int]:
   if len(set1) >= len(set2):
      reps = list(set1) + [-r for r in set2]
   else:
      reps = list(set2) + [-r for r in set1]
   e, n = encalc_times_36(reps, repinfo)
   rew_row = [e, n] + reps
   return rew_row

def create_full_catalogue(nq_max: int, verbose: bool = True) -> None:
   if nq_max < 2:
      raise ValueError("nq_max must be at least 2.")
   t0 = time.time()
   if verbose:
      qrange = trange(2, nq_max+1)
   else:
      qrange = range(2, nq_max+1)
   for q in qrange:
      t1 = time.time()
      previous_cats = ["output/data/"+f for f in os.listdir("output/data/") if f.startswith(f"addNQ{q:02d}_m")]
      n_masses = len(previous_cats)
      n_models = 0
      for h5name_old in previous_cats:
         s = h5name_old.split("add")
         h5name_new = s[0] + "all" + s[1]
         with h5.File(h5name_old, 'r') as f:
            mQ = f.attrs['m_Q']
            lp_threshold = f.attrs['LP_threshold']
            evals = f["E"][:]
            nvals = f["N"][:]
            models = f["model"][:]
            if verbose:
               print(f"Reading file {h5name_old} with {len(models):d} models to extend", flush=True)
         data = np.column_stack((evals, nvals, models))
         if len(models) > 0:
            for mod in models:
               # N.B. The multiset_partitions generator expects a list as the first argument
               new_rows = []
               for set1,set2 in multiset_partitions(list(mod),2):
                  new_row = extend_all_models(set1, set2, repinfo)
                  new_rows.append(new_row)
               new_rows = np.array(new_rows, dtype='int')
               data = np.vstack((data, new_rows))
               n_models += len(new_rows)
            with h5.File(h5name_new, 'w') as f:
               f.attrs['LP_threshold'] = lp_threshold
               f.attrs['m_Q'] = mQ
               f.create_dataset("model", data=data[:,2:], dtype='i2')
               f.create_dataset("E", data=data[:,0], dtype='i4')
               f.create_dataset("N", data=data[:,1], dtype='i4')
         elif verbose:
            print(f"No models to extend for N_Q = {q:d} and mass {mQ:.2e} GeV.", flush=True)
      print("Computed {:d} additional models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(n_models, q, n_masses, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)