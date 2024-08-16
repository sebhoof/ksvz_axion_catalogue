# create_catalogues.py

import h5py as h5
import numpy as np
import os.path
import time

from itertools import combinations_with_replacement
from numba import njit
from tqdm import trange

from .model_building import encalc_times_36, find_LP, repinfo

@njit("int64[:,:](int64[:,:],int64[:,:])")
def compute_eon_values(models: np.ndarray[int], repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    eonvals = []
    for reps in models:
        e, n = encalc_times_36(reps, repinfo)
        eonvals.append([e, n])
    return np.array(eonvals, dtype='int')

def save_initial_catalogue(masses: list[float], reps: list[int]):
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
def extend_models(models: np.ndarray[int]) -> np.ndarray[int]:
    new_models = [[i]+list(mod) for mod in models for i in range(1, mod[0]+1)]
    return np.array(new_models, dtype='int')

def create_extended_catalogue(nq_max: int, lp_threshold: float = 1.0e18, verbose: bool = True):
    if nq_max < 3:
        raise ValueError("N_Q must be at least 3.")
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
                new_models = extend_models(models_to_extend)
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
        print("Computed {:d} models for N_Q = {:d} with {:d} mass(es) after {:.2f} mins.".format(len(new_models), q, n_masses, (time.time()-t1)/60), flush=True)
    print("All tasks completed after {:.2f} mins.".format((time.time()-t0)/60), flush=True)
        