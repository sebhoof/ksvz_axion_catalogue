# create_catalogues.py

import h5py as h5
import numpy as np
import os.path
# import pickle
import time

# from itertools import combinations_with_replacement
from numba import njit
from sympy.utilities.iterables import multiset_partitions

from .constants import PLANCK18_OMH2_LIMIT
from .cosmo import *
from .model_building import encalc_times_36, find_LP, min_dim_from_rep, repinfo

@njit
def compute_eon_values(models: np.ndarray[int], repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
   """
   A wrapper function to compute the E and N values for a list of models.

   Parameters
   ----------
   models : np.ndarray[int]
      An (m,n)-array of m models containing the n representations of the Qs
   repinfo : np.ndarray[int]
      A 2d array containing the information about the representations

   Returns
   -------
   np.ndarray[int]
      An (m,2)-array of 36*E and 36*N integer values for each model
   """
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
   """
   Helper function to append data to an HDF5 file.
   """
   file.create_group(gr_str)
   file[gr_str].create_dataset("model", data=mods, dtype='i2')
   file[gr_str].create_dataset("E", data=evals, dtype='i4')
   file[gr_str].create_dataset("N", data=nvals, dtype='i4')
   if len(lps) > 0:
      file[gr_str].create_dataset("LP", data=lps, dtype='f8')

@njit
def extend_all_models(set1: list[int], set2: list[int], repinfo: np.ndarray[int] = repinfo) -> list[int]:
   if len(set1) >= len(set2):
      reps = list(set1) + [-r for r in set2]
   else:
      reps = list(set2) + [-r for r in set1]
   e, n = encalc_times_36(reps, repinfo)
   rew_row = [e, n] + reps
   return rew_row

"""
def check_additive_model_fast(models, mQ, mindex, cosmo_dict, w_threshold: float, lp_threshold: float, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, output_root: str = "output/catalogues/", verbose: bool = False) -> np.ndarray[int]:
   omh2_std = omh2_axion_sm(mQ, thetai)
   # wBBN_max, omh2_min = 0, np.inf
   new_models = []
   for m in models:
      wBBN, omh2 = EOS_SM_BBN, omh2_std
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
                  ubreaks, tebreaks, sols, dims, mults = compute_cosmology(qdims, qmult, mQ, verbose=False)
                  mult_string = dim_signature(qdims, qmult)
                  output_file = output_root+"alt_cosmo"+mult_string+f"_m{mindex:d}.dat"
                  wBBN, _ = save_cosmology(ubreaks, tebreaks, sols, dims, mults, mQ, output_file, plot=False)
                  omh2 = omh2_axion(mQ, output_file, thetai)
                  cosmo_dict[tpl_dims] = (wBBN, omh2)
                  if verbose:
                     print(f"New case: {tpl_dims} | wBBN = {wBBN:.4f}, omh2 = {omh2:.2e}.", flush=True)
         bbn_check = (wBBN > w_threshold)
         omh2_check = (omh2 < omh2_threshold)
         if bbn_check and omh2_check:
            new_models.append(m)
   return np.array(new_models, dtype='int')
"""

def check_additive_models(models: list[list[int]], mQ: float, mindex: int, w_threshold: float, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, eft_scale: float = M_PLANCK, output_root: str = "output/catalogues/", cosmo_dict: dict = {}, verbose: bool = False) -> np.ndarray[int]:
   valid_additive_models = []
   omh2_std = omh2_axion_sm(mQ, thetai)
   wBBN_max, omh2_min = 0, np.inf
   for m in models:
      wBBN, omh2 = EOS_SM_BBN, omh2_std
      dims = np.array([min_dim_from_rep(r) for r in m], dtype='int')
      # Don't have expressions for d = 3 operators, so use d = 4 (inconsequential)
      d_max = max(4, max(dims))
      std_cond = gammad(mQ, d=d_max, scale=eft_scale) > 100*H_QCD
      if not(std_cond) and (d_max > 4):
         qdims, qmult = np.unique(dims[dims > 4], return_counts=True)
         mult_string = dim_signature(qdims, qmult)
         if mult_string in cosmo_dict:
            wBBN, omh2 = cosmo_dict[mult_string]
         else:
            ubreaks, tebreaks, sols, dims, mults = compute_cosmology(qdims, qmult, mQ, verbose=False)
            output_file = output_root+"alt_cosmo"+mult_string+f"_m{mindex:d}.dat"
            wBBN, _ = save_cosmology(ubreaks, tebreaks, sols, dims, mults, mQ, output_file, plot=False)
            omh2 = omh2_axion(mQ, output_file, thetai)
            cosmo_dict[mult_string] = (wBBN, omh2)
            if verbose:
               print(f"New case: {mult_string:s} | wBBN = {wBBN:.4f}, omh2 = {omh2:.2e} (omh2_std = {omh2_std:.2e}).", flush=True)
      wBBN_max = max(wBBN_max, wBBN)
      bbn_check = (wBBN > w_threshold)
      omh2_min = min(omh2_min, omh2)
      omh2_check = (omh2 < omh2_threshold)
      if bbn_check and omh2_check:
         valid_additive_models.append(m)
   return np.array(valid_additive_models, dtype='int'), wBBN_max, omh2_min

@njit
def extend_all_models_fast(set1: list[int], set2: list[int]) -> list[int]:
   if len(set1) >= len(set2):
      reps = list(set1) + [-r for r in set2]
   else:
      reps = list(set2) + [-r for r in set1]
   return reps

"""
def additive_catalogues_fast(masses: list[float], reps: list[int], lp_threshold: float = 1.0e18, w_threshold: float = 0.3, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, output_root: str = "", verbose: bool = False) -> None:
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
         models_to_add = check_additive_model_fast(new_models, mQ, i, cosmo_dict, w_threshold, lp_threshold, omh2_threshold, thetai=thetai, output_root=output_root, verbose=verbose)
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
         f.attrs["EOS_threshold"] = w_threshold
         f.attrs["Omh2_threshold"] = omh2_threshold
         for j,mods in enumerate(additive_models):
            gr_str = f"NQ{(j+1):d}"
            f.create_group(gr_str)
            f[gr_str].create_dataset("model", data=mods, dtype='i2')
      with open(cosmo_fname, 'wb') as g:
         pickle.dump(cosmo_dict, g, protocol=pickle.HIGHEST_PROTOCOL)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)
"""

def check_hdf5_file(h5name: str, mQ: float, lp_threshold: float, w_threshold: float, omh2_threshold: float) -> bool:
   checks = 0
   if os.path.isfile(h5name):
      with h5.File(h5name, 'r') as f:
         checks += (f.attrs["m_Q"] == mQ)
         checks += (f.attrs["LP_threshold"] == lp_threshold)
         checks += (f.attrs["EOS_threshold"] == w_threshold)
         checks += (f.attrs["Omh2_threshold"] == omh2_threshold)
   return checks == 4

def additive_catalogues(masses: list[float], reps: list[int], lp_threshold: float = 1.0e18, w_threshold: float = 0.3, omh2_threshold: float = PLANCK18_OMH2_LIMIT, thetai: float = 2.2, eft_scale: float = M_PLANCK, output_root: str = "", verbose: bool = False) -> None:
   t0 = time.time()
   wBBN_max, omh2_min = 0, np.inf
   for mindex,mQ in enumerate(masses):
      omh2_std = omh2_axion_sm(mQ, thetai)
      nQ = 1
      base_models = []
      h5name = output_root+f"small_add_KSVZ_models_m{mindex:d}.h5"
      if output_root != "":
         if check_hdf5_file(h5name, mQ, lp_threshold, w_threshold, omh2_threshold):
            if verbose:
               print(f"File {h5name} already exists for m_Q = {mQ:.2e} GeV. Skipping...", flush=True)
            continue
      extend_models = True
      while extend_models:
         t1 = time.time()
         if nQ == 1:
            new_models = np.array(reps, dtype='int')[:,np.newaxis]
         else:
            models_to_extend = base_models[nQ-2]
            new_models = extend_additive_models(models_to_extend, base_models[0].T[0])
         # Only perform first two checks for combined models
         if nQ > 1:
            # First, remove models that are clearly matter-dominated at BBN
            max_dims = np.maximum(4, [max([min_dim_from_rep(r) for r in m]) for m in new_models])
            md_cond = np.array([gammad(mQ, d=dim, scale=eft_scale) for dim in max_dims]) > 0.01*H_BBN
            new_models = new_models[md_cond]
            # Second, if the standard cosmology is excluded, exclude models that induce standard cosmology
            if omh2_std > omh2_threshold:
               max_dims = max_dims[md_cond]
               rd_cond = np.array([gammad(mQ, d=dim, scale=eft_scale) for dim in max_dims]) < 100*H_QCD
               new_models = new_models[rd_cond]
         # Third, remove models that induce a LP below the threshold
         lp_cond = np.array([find_LP(m, mQ, lp_threshold, verbose=False, plot=False)[0] for m in new_models]) > lp_threshold
         new_models = new_models[lp_cond]
         n_allowed_models = len(new_models)
         if n_allowed_models > 0:
            base_models.append(new_models)
            if verbose:
               print(f"m_Q = {mQ:.2e} GeV, N_Q = {nQ:d}: {n_allowed_models:d} potentially allowed models.", flush=True)
         else:
            extend_models = False
            if verbose:
               print(f"m_Q = {mQ:.2e} GeV, N_Q = {nQ:d}: no more potentially allowed models.", flush=True)
         nQ += 1
      additive_models, n_valid_models = [], 0
      cosmo_dict = {}
      for models in base_models:
         valid_models, wBBN, omh2 = check_additive_models(models, mQ, mindex, w_threshold, omh2_threshold, thetai, eft_scale, output_root, cosmo_dict, verbose)
         n_valid_models += len(valid_models)
         additive_models.append(valid_models)
         wBBN_max = max(wBBN, wBBN_max)
         omh2_min = min(omh2, omh2_min)
      if output_root != "":
         with h5.File(h5name, 'w') as f:
            f.attrs["m_Q"] = mQ
            f.attrs["LP_threshold"] = lp_threshold
            f.attrs["EOS_threshold"] = w_threshold
            f.attrs["Omh2_threshold"] = omh2_threshold
            for nQm1,models in enumerate(additive_models):
               gr_str = f"NQ{(nQm1+1):d}"
               f.create_group(f"NQ{(nQm1+1):d}")
               f[gr_str].create_dataset("model", data=models, dtype='i2')
      print("Found {:d} valid additive models for m_Q = {:.2e} GeV and N_Q < {:d} after {:.2f} mins.".format(n_valid_models, mQ, nQ, (time.time()-t1)/60), flush=True)
   if verbose:
      print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)
   return wBBN_max, omh2_min

def full_catalogues_from_additive_catalogues(mass_indices: list[int], output_root: str = "output/catalogues/") -> None:
   t0 = time.time()
   for i in mass_indices:
      h5name = output_root+f"small_KSVZ_models_m{i:d}.h5"
      if os.path.isfile(h5name):
         print(f"File {h5name} already exists. Skipping...", flush=True)
         continue
      h5name_add = output_root+f"small_add_KSVZ_models_m{i:d}.h5"
      if not os.path.isfile(h5name_add):
         raise RuntimeError(f"File {h5name_add} does not exist.")
      groups = []
      with h5.File(h5name_add, 'r') as f0:
         mQ = f0.attrs["m_Q"]
         lp_threshold = f0.attrs["LP_threshold"]
         w_threshold = f0.attrs["EOS_threshold"]
         omh2_threshold = f0.attrs["Omh2_threshold"]
         groups = list(f0.keys())
      with h5.File(h5name, 'w') as f1:
         f1.attrs["m_Q"] = mQ
         f1.attrs["LP_threshold"] = lp_threshold
         f1.attrs["EOS_threshold"] = w_threshold
         f1.attrs["Omh2_threshold"] = omh2_threshold
      for gr in groups:
         t1 = time.time()
         with h5.File(h5name_add, 'r') as f0:
            additive_models = f0[gr]["model"][:]
         n_new_models = 0
         nQ = int(gr[2:])
         if nQ == 1:
            eonvals = compute_eon_values(additive_models, repinfo)
            data = np.column_stack((eonvals, additive_models.flatten()))
         else:
            data = np.empty((0,nQ+2), dtype='int')
            for mod in additive_models:
               new_rows = np.array([extend_all_models(mod, [], repinfo)], dtype='int')
               data = np.vstack((data, new_rows))
               # N.B. multiset_partitions() expects a list as the first argument
               new_rows = np.array([extend_all_models(set1, set2, repinfo) for set1,set2 in multiset_partitions(list(mod),2)], dtype='int')
               n_new_models += len(new_rows)
               data = np.vstack((data, new_rows))
         with h5.File(h5name, 'a') as f1:
            append_data(f1, gr, data[:,2:], data[:,0], data[:,1])
         print("m_Q = {:.2e} GeV, N_Q = {:d}: computed {:d} new models after {:.2f} mins.".format(mQ, nQ, n_new_models, (time.time()-t1)/60), flush=True)
   print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)