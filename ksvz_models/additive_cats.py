# additive_cats.py

import h5py as h5
import numpy as np
import time
# import warnings

from itertools import combinations_with_replacement
from fractions import Fraction
from numba import njit
from tqdm import trange

from .functionsfile import encalc_times_36, find_LP, repinfo

# warnings.filterwarnings("ignore", category=RuntimeWarning)

def create_initial_catalogue(models: list[list[int]]) -> list[list[int]]:
    eonvals = []
    for reps in models:
        e, n = encalc_times_36(reps, repinfo)
        eonvals.append([e, n])
    return eonvals

def save_initial_catalogue(masses: list[float], reps: list[int]):
    for q in trange(1,3):
        t0 = time.time()
        models = list(combinations_with_replacement(reps, q))
        print(f"Created combinations for N_Q = {q:d}")
        eonvals = create_initial_catalogue(models)
        eonvals = [(Fraction(i[0], 36), Fraction(i[1], 36)) for i in eonvals]
        eonvals = np.array([[e.numerator, e.denominator, n.numerator, n.denominator] for e,n in eonvals], dtype='int')
        with h5.File(f"output/data/addNQ{q:d}.h5", 'w') as f:
            f.create_dataset("E_numerator", data=eonvals[:,0], dtype='i2')
            f.create_dataset("E_denominator", data=eonvals[:,1], dtype='i2')
            f.create_dataset("N_numerator", data=eonvals[:,2], dtype='i2')
            f.create_dataset("N_denominator", data=eonvals[:,3], dtype='i2')
            f.create_dataset("model", data=models, dtype='i2')
        for mQ in masses:
            lps = []
            for model in models:
                lp, _ = find_LP(model, mQ, plot=False)
                lps.append(lp)
            with h5.File(f"output/data/addNQ{q:d}.h5", 'a') as f:
                f.create_dataset(f"LP_m{int(mQ/1e7):d}", data=lps, dtype='f8')
        print(f"All done for NQ = {q:d} after {(time.time()-t0):.2f} mins.")