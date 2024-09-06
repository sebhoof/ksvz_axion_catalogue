# create_catalogues.py

import h5py as h5
import numpy as np
import os.path
import time

from itertools import combinations_with_replacement
from numba import njit
from sympy.utilities.iterables import multiset_partitions

from .model_building import encalc_times_36, find_LP, repinfo

@njit
def compute_eon_values(models: np.ndarray[int], repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
   eonvals = []
   for reps in models:
      e, n = encalc_times_36(reps, repinfo)
      eonvals.append([e, n])
   return np.array(eonvals, dtype='int')

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

def append_data(file, gr_str: str, mods: np.ndarray[int], evals: np.ndarray[int], nvals: np.ndarray[int], lps: np.ndarray[float] = []) -> None:
   file.create_group(gr_str)
   file[gr_str].create_dataset("model", data=mods, dtype='i2')
   file[gr_str].create_dataset("E", data=evals, dtype='i4')
   file[gr_str].create_dataset("N", data=nvals, dtype='i4')
   if len(lps) > 0:
      file[gr_str].create_dataset("LP", data=lps, dtype='f8')

def additive_catalogues(masses: list[float], reps: list[int], nq_max: int = 31, lp_threshold: float = 1.0e18) -> None:
   t0 = time.time()
   for i,mQ in enumerate(masses):
      h5name = f"output/data/add_KSVZ_models_m{i:d}.h5"
      if os.path.isfile(h5name):
         with h5.File(h5name, 'r') as f:
            if (f.attrs["LP_threshold"] != lp_threshold) or (f.attrs["m_Q"] != mQ):
               raise RuntimeError(f"File {h5name} already exists but with different mQ or LP threshold.")
      with h5.File(h5name, 'a') as f:
         extend_models = True
         f.attrs["LP_threshold"] = lp_threshold
         f.attrs["m_Q"] = mQ
         for nQ in range(1, nq_max+1):
            t1 = time.time()
            gr_str = f"NQ{nQ:02d}"
            if gr_str in f:
               print(f"Group {gr_str} already exists in file {h5name}; skipping...", flush=True)
               continue
            if nQ < 3:
               models = np.array(list(combinations_with_replacement(reps, nQ)), dtype='int')
               eonvals = compute_eon_values(models, repinfo)
               lps = np.array([find_LP(model, mQ, plot=False)[0] for model in models])
               cond = (lps >= lp_threshold)
               n_valid_models = sum(cond)
               append_data(f, gr_str, models[cond], eonvals[cond,0], eonvals[cond,1], lps[cond])
            elif extend_models:
               models_to_extend = f[f"NQ{(nQ-1):02d}"]["model"][:]
               allowed_reps = f["NQ01"]["model"][:].T[0]
               new_models = extend_additive_models(models_to_extend, allowed_reps)
               eonvals = compute_eon_values(new_models, repinfo)
               lps = np.array([find_LP(model, mQ, plot=False)[0] for model in new_models])
               cond = (lps >= lp_threshold)
               n_valid_models = sum(cond)
               if n_valid_models > 0:
                  append_data(f, gr_str, new_models[cond], eonvals[cond,0], eonvals[cond,1], lps[cond])
               else:
                  extend_models = False
            else:
               print(f"No models to extend for m_Q = {mQ:.2e} GeV and N_Q = {nQ:d}.", flush=True)
               continue
            print("Computed {:d} valid models for m_Q = {:.2e} GeV and N_Q = {:d} after {:.2f} mins.".format(n_valid_models, mQ, nQ, (time.time()-t1)/60), flush=True)
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

def create_full_catalogues(nq_max: int) -> None:
   t0 = time.time()
   mass_indices = [int(f.split("models_m")[1].split(".h5")[0]) for f in os.listdir("output/data/") if f.startswith("add_KSVZ")]
   for i in mass_indices:
      h5name = f"output/data/KSVZ_models_m{i:d}.h5"
      with h5.File(h5name, 'a') as f:
         h5name_add = f"output/data/add_KSVZ_models_m{i:d}.h5"
         with h5.File(h5name_add, 'r') as f0:
            mQ = f0.attrs['m_Q']
            lp_threshold = f0.attrs['LP_threshold']
            f.attrs['LP_threshold'] = lp_threshold
            f.attrs['m_Q'] = mQ
            for nQ in range(1, nq_max+1):
               gr_str = f"NQ{nQ:02d}"
               if gr_str in f:
                  print(f"Group {gr_str} already exists in file {h5name}; skipping...", flush=True)
                  continue
               if not gr_str in f0:
                  print(f"Group {gr_str} does not exist in file {h5name_add}; skipping...", flush=True)
                  continue
               if nQ == 1:
                  append_data(f, "NQ01", f0["NQ01"]["model"][:], f0["NQ01"]["E"][:], f0["NQ01"]["N"][:])
                  continue
               t1 = time.time()
               evals = f0[gr_str]["E"][:]
               nvals = f0[gr_str]["N"][:]
               models = f0[gr_str]["model"][:]
               data = np.column_stack((evals, nvals, models))
               n_new_models = 0
               for mod in models:
                  # N.B. multiset_partitions() expects a list as the first argument
                  new_rows = np.array([extend_all_models(set1, set2, repinfo) for set1,set2 in multiset_partitions(list(mod),2)], dtype='int')
                  n_new_models += len(new_rows)
                  data = np.vstack((data, new_rows))
               append_data(f, gr_str, data[:,2:], data[:,0], data[:,1])
               print("Computed {:d} new models for m_Q = {:.2e} GeV and N_Q = {:d} after {:.2f} mins.".format(n_new_models, mQ, nQ, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)