import numpy as np
from itertools import combinations_with_replacement
import h5py as h5
import time
import warnings
from tqdm import tqdm
from functionsfile import *

threshold = 10**18
#masses = 10**7*np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]).astype(float)
#masses = 10**7*np.array([1000000]).astype(float)
masses = [5*10**11];

warnings.filterwarnings("ignore", category=RuntimeWarning)

reps = np.linspace(1,112,112).astype(int)

for m in masses:
    start=time.time()
    for q in tqdm(range(1,3)):
        e_num, e_den, n_num, n_den, LP, mod = [],[],[],[],[],[]
        models = list(combinations_with_replacement(reps,q))
        print("Created combinations for NQ =",q)
        for i in tqdm(models):
            temp=[]
            for j in list(i):
                temp.append(repdict[j])
            e_num.append((Fraction(ENcalc(temp)[0]).limit_denominator()).numerator)
            e_den.append((Fraction(ENcalc(temp)[0]).limit_denominator()).denominator)
            n_num.append((ENcalc(temp)[1]).numerator)
            n_den.append((ENcalc(temp)[1]).denominator)
            f=[np.array(temp)[:,2], np.array(temp)[:,1], np.array(temp)[:,0]]
            l1,l2,l3=do_it(f, m)
            LP.append(l1)
            mod.append(list(i))
        with h5.File('output/data/mass{}/addNQ{}.h5'.format(int(m/10**7),q), 'w') as f:
            f.create_dataset("E_numerator", data=e_num, dtype='i4')
            f.create_dataset("E_denominator", data=e_den, dtype='i4')
            f.create_dataset("N_numerator", data=n_num, dtype='i4')
            f.create_dataset("N_denominator", data=n_den, dtype='i4')
            f.create_dataset("LP", data=LP, dtype='f8')
            f.create_dataset("model", data=mod)
        print("NQ =",q," done.")
    print('Mass {} done in {} minutes'.format(m, (time.time()-start)/60))