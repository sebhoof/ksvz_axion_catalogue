# create_catalogues.py

import h5py as h5
import numpy as np
import os.path
import pickle
import time

from itertools import combinations_with_replacement
from numba import njit
from sympy.utilities.iterables import multiset_partitions

from .constants import M_PLANCK_RED, PLANCK18_OMH2_LIMIT
from .cosmo import compute_cosmology, omh2_axion, omh2_axion_sm, save_cosmology
from .model_building import encalc_times_36, find_LP, min_dim_from_rep, repinfo

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
            gr_str = f"NQ{nQ:d}"
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
               models_to_extend = f[f"NQ{(nQ-1):d}"]["model"][:]
               allowed_reps = f["NQ1"]["model"][:].T[0]
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

def create_full_catalogues(mass_indices: np.ndarray[int], nq_max: int) -> None:
   t0 = time.time()
   # mass_indices = [int(f.split("models_m")[1].split(".h5")[0]) for f in os.listdir("output/data/") if f.startswith("add_KSVZ")]
   for i in mass_indices:
      h5name = f"output/data/KSVZ_models_m{i:d}.h5"
      h5name_add = f"output/data/add_KSVZ_models_m{i:d}.h5"
      if not os.path.isfile(h5name_add):
         raise RuntimeError(f"File {h5name_add} does not exist.", flush=True)
      with h5.File(h5name, 'a') as f:
         with h5.File(h5name_add, 'r') as f0:
            mQ = f0.attrs['m_Q']
            lp_threshold = f0.attrs['LP_threshold']
            f.attrs['LP_threshold'] = lp_threshold
            f.attrs['m_Q'] = mQ
            for nQ in range(1, nq_max+1):
               gr_str = f"NQ{nQ:d}"
               if gr_str in f:
                  print(f"Group {gr_str} already exists in file {h5name}; skipping...", flush=True)
                  continue
               if not gr_str in f0:
                  print(f"Group {gr_str} does not exist in file {h5name_add}; skipping...", flush=True)
                  continue
               if nQ == 1:
                  append_data(f, "NQ1", f0["NQ1"]["model"][:], f0["NQ1"]["E"][:], f0["NQ1"]["N"][:])
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

def check_additive_model_fast(models, mQ, mindex, cosmo_dict, bbn_threshold: float, lp_threshold: float, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, output_root: str = "output/catalogues/", verbose: bool = False) -> None:
   omh2_std = omh2_axion_sm(mQ, thetai)
   new_models = []
   for m in models:
      wBBN, omh2 = 1, omh2_std
      lp, _ = find_LP(m, mQ, plot=False)
      lp_check = (lp > lp_threshold)
      if lp_check:
         dims = np.array([min_dim_from_rep(r) for r in m], dtype='int')
         dims = dims[dims > 4]
         qdims, qmult = np.unique(dims, return_counts=True)
         if len(qdims) > 0:
            # Safely rule out scenarios with high-dimensional representations and low mQ
            if (sum(dims > 5) > 0) and (mQ <= 1e10):
               wBBN = 0
               omh2 = np.nan
            # If first check fails, then we have d = 5 operators and standard cosmology -- unless mQ is high enough
            elif (mQ > 1e10):
               tpl_dims = tuple([qmult[qdims==d0][0] if d0 in qdims else 0 for d0 in range(5,9)])
               try:
                  wBBN, omh2 = cosmo_dict[tpl_dims]
               except KeyError:
                  ubreaks, tebreaks, sols, mult_string = compute_cosmology(qdims, qmult, mQ, verbose=False)
                  output_file = output_root+"alt_cosmo"+mult_string+f"_m{mindex:d}.dat"
                  wBBN, _ = save_cosmology(ubreaks, tebreaks, sols, mQ, output_file, plot=False)
                  omh2 = omh2_axion(mQ, output_file, thetai)
                  cosmo_dict[tpl_dims] = (wBBN, omh2)
                  if verbose:
                     print(f"New case: {tpl_dims} | wBBN = {wBBN:.4f}, omh2 = {omh2:.2e}.", flush=True)
         bbn_check = (wBBN > bbn_threshold)
         omh2_check = (omh2 < omh2_threshold)
         if bbn_check and omh2_check:
            new_models.append(m)
   return np.array(new_models, dtype='int')

@njit
def extend_all_models_fast(set1: list[int], set2: list[int]) -> list[int]:
   if len(set1) >= len(set2):
      reps = list(set1) + [-r for r in set2]
   else:
      reps = list(set2) + [-r for r in set1]
   return reps

def additive_catalogues_fast(masses: list[float], reps: list[int], lp_threshold: float = 1.0e18, bbn_threshold: float = 0.9, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, output_root: str = "output/catalogues/", verbose: bool = False) -> None:
   t0 = time.time()
   for i,mQ in enumerate(masses):
      nQ = 1
      additive_models = []
      h5name = output_root+f"small_add_KSVZ_models_m{i:d}.h5"
      cosmo_fname = output_root+f"cosmo_dict_m{i:d}.pkl"
      cosmo_dict = {}
      if os.path.isfile(cosmo_fname):
         with open(cosmo_fname, 'rb') as file:
            cosmo_dict = pickle.load(file)
      extend_models = True
      while extend_models:
         t1 = time.time()
         if nQ < 3:
            new_models = np.array(list(combinations_with_replacement(reps, nQ)), dtype='int')
         else:
            models_to_extend = additive_models[nQ-2]
            new_models = extend_additive_models(models_to_extend, additive_models[0].T[0])
         models_to_add = check_additive_model_fast(new_models, mQ, i, cosmo_dict, bbn_threshold, lp_threshold, omh2_threshold, thetai=thetai, output_root=output_root, verbose=verbose)
         n_valid_models = len(models_to_add)
         if verbose:
            print("Computed {:d} valid models for m_Q = {:.2e} GeV and N_Q = {:d} after {:.2f} mins.".format(n_valid_models, mQ, nQ, (time.time()-t1)/60), flush=True)
         if n_valid_models > 0:
            additive_models.append(models_to_add)
         else:
            extend_models = False
         nQ += 1
      with h5.File(h5name, 'w') as f:
         f.attrs["m_Q"] = mQ
         f.attrs["LP_threshold"] = lp_threshold
         f.attrs["BBN_threshold"] = bbn_threshold
         f.attrs["Omh2_threshold"] = omh2_threshold
         for j,mods in enumerate(additive_models):
            gr_str = f"NQ{(j+1):d}"
            f.create_group(gr_str)
            f[gr_str].create_dataset("model", data=mods, dtype='i2')
      with open(cosmo_fname, 'wb') as g:
         pickle.dump(cosmo_dict, g, protocol=pickle.HIGHEST_PROTOCOL)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)

def full_catalogues_fast(mass_indices: list[int], output_root: str = "output/catalogues/") -> None:
   t0 = time.time()
   # mass_indices = [int(f.split("models_m")[1].split(".h5")[0]) for f in os.listdir(out_dir) if f.startswith("small_add_KSVZ")]
   for i in mass_indices:
      h5name = output_root+f"small_KSVZ_models_m{i:d}.h5"
      h5name_add = output_root+f"small_add_KSVZ_models_m{i:d}.h5"
      if not os.path.isfile(h5name_add):
         raise RuntimeError(f"File {h5name_add} does not exist.")
      groups = []
      with h5.File(h5name_add, 'r') as f0:
         mQ = f0.attrs["m_Q"]
         lp_threshold = f0.attrs["LP_threshold"]
         bbn_threshold = f0.attrs["BBN_threshold"]
         omh2_threshold = f0.attrs["Omh2_threshold"]
         groups = list(f0.keys())
      with h5.File(h5name, 'w') as f1:
         f1.attrs["m_Q"] = mQ
         f1.attrs["LP_threshold"] = lp_threshold
         f1.attrs["BBN_threshold"] = bbn_threshold
         f1.attrs["Omh2_threshold"] = omh2_threshold
      for gr in groups:
         t1 = time.time()
         with h5.File(h5name_add, 'r') as f0:
            additive_models = f0[gr]["model"][:]
         n_new_models = 0
         nQ = int(gr[2:])
         if (nQ == 1):
            eonvals = compute_eon_values(additive_models, repinfo)
            data = np.column_stack((eonvals, additive_models.flatten()))
         else:
            data = np.empty((0,nQ+2), dtype='int')
            for mod in additive_models:
               # N.B. multiset_partitions() expects a list as the first argument
               new_rows = np.array([extend_all_models(set1, set2, repinfo) for set1,set2 in multiset_partitions(list(mod),2)], dtype='int')
               n_new_models += len(new_rows)
               data = np.vstack((data, new_rows))
         with h5.File(h5name, 'a') as f1:
            append_data(f1, gr, data[:,2:], data[:,0], data[:,1])
         print("Computed {:d} new models for m_Q = {:.2e} GeV (N_Q = {:d}) after {:.2f} mins.".format(n_new_models, mQ, nQ, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)

def check_additive_model_root_finding(mQ: float, models, lp_threshold: float, bbn_threshold: float = 0.9, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, verbose: bool = False):
   if mQ < 1e9:
      raise ValueError("mQ must be at least 1e9 GeV for now.")
   # omh2_std = omh2_axion_sm(mQ, thetai)
   wBBN_max, omh2_min = 0, np.inf
   models_to_add = []
   known_cosmo = []
   for m in models:
      lp = find_LP(m, mQ, plot=False)[0]
      lp_check = (lp > lp_threshold)
      if lp_check:
         dims = np.array([min_dim_from_rep(r) for r in m], dtype='int')
         dims = dims[dims > 4]
         qdims, qmult = np.unique(dims, return_counts=True)
         tpl_dims = tuple([qmult[qdims==d0][0] if d0 in qdims else 0 for d0 in range(5,9)])
         has_high_d_mod = len(qdims) > 0
         is_new_cosmo = tpl_dims not in known_cosmo
         if is_new_cosmo and has_high_d_mod:
            known_cosmo.append(tpl_dims)
            ubreaks, tebreaks, sols, mult_string = compute_cosmology(qdims, qmult, mQ, verbose=verbose)
            output_file = "/Users/sebhoof/Software/ksvz_axion_catalogue/output/cosmo/alt_cosmo"+mult_string+".dat"
            wBBN, _ = save_cosmology(ubreaks, tebreaks, sols, mQ, output_file, plot=False, verbose=False)
            bbn_check = (wBBN > bbn_threshold)
            omh2 = omh2_axion(mQ, output_file, thetai)
            omh2_check = (omh2 < omh2_threshold)
            if omh2_check:
               wBBN_max = max(wBBN, wBBN_max)
            if bbn_check:
               omh2_min = min(omh2, omh2_min)
            if bbn_check:
               models_to_add.append(m)
               if verbose:
                  print(f"New case: {tpl_dims} | wBBn = {wBBN:.4f}, omh2 = {omh2:.2e}.", flush=True)
   return models_to_add, wBBN_max, omh2_min

def models_for_root_finding(mQ, reps: list[int], nq_max: int = 31, lp_threshold: float = 1.0e18, bbn_threshold: float = 0.9, omh2_threshold: float = PLANCK18_OMH2_LIMIT, verbose: bool = False) -> None:
   t0 = time.time()
   wBBN_max, omh2_min = 0, np.inf
   catalogue_data = []
   extend_models = True
   for nQ in range(1, nq_max+1):
      t1 = time.time()
      if nQ < 3:
         models = np.array(list(combinations_with_replacement(reps, nQ)), dtype='int')
      elif extend_models:
         models_to_extend = catalogue_data[nQ-2]
         allowed_reps = np.array(catalogue_data[0], dtype='int').T[0]
         models = extend_additive_models(models_to_extend, allowed_reps)
      else:
         if verbose:
            print("No models to extend for m_Q = {:.2e} GeV and N_Q = {:d}.".format(mQ, nQ), flush=True)
         continue
      models_to_add, wBBN, omh2 = check_additive_model_root_finding(mQ, models, lp_threshold, bbn_threshold, omh2_threshold, verbose=verbose)
      wBBN_max = max(wBBN, wBBN_max)
      omh2_min = min(omh2, omh2_min)
      n_valid_models = len(models_to_add)
      print("Computed {:d} valid models for m_Q = {:.2e} GeV and N_Q = {:d} after {:.2f} mins.".format(n_valid_models, mQ, nQ, (time.time()-t1)/60), flush=True)
      if n_valid_models > 0:
         catalogue_data.append(models_to_add)
      else:
         extend_models = False
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)
   return wBBN_max, omh2_min