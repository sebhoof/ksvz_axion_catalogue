import h5py as h5
import numpy as np
from functionsfile import *
import time
from sympy.utilities.iterables import multiset_partitions
from tqdm import tqdm

#masses = 10**7*np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]).astype(float)
#masses = 10**7*np.array([1000000]).astype(float)
masses = [5*10**11];

reps = np.linspace(1,112,112).astype(int)

threshold = 10**18

for mass in masses:
    start = time.time()
    for q in range(15,51):
        e_num, e_den, n_num, n_den, new_mod = [],[],[],[],[]
        try:
            with h5.File('output/data/mass{}/addNQ{}.h5'.format(int(mass/10**7),q), 'r') as f:
                prev_lp = f['LP'][:]
                prev_mod = f['model'][:]
                e_num = f['E_numerator'][:]
                e_den = f['E_denominator'][:]
                n_num = f['N_numerator'][:]
                n_den = f['N_denominator'][:]
        except:
            break
        cond = prev_lp >= threshold
        mods_to_ext = prev_mod[cond]
        new_mod = list(mods_to_ext)
        e_num = list(e_num[cond])
        e_den = list(e_den[cond])
        n_num = list(n_num[cond])
        n_den = list(n_den[cond])
#        del(prev_mod)
#        del(prev_lp)
        if len(mods_to_ext)==0:
            break
        for mod in tqdm(mods_to_ext):
            temp = list(multiset_partitions(list(mod),2))
            for j in temp:
                if len(j[0])>=len(j[1]):
                    repsum = [repdict[k] for k in j[0]]
                    repsub = [repdict[k] for k in j[1]]
                else:
                    repsum = [repdict[k] for k in j[1]]
                    repsub = [repdict[k] for k in j[0]]
                e_num.append((Fraction(ENcalc(temp)[0]).limit_denominator()).numerator)
                e_den.append((Fraction(ENcalc(temp)[0]).limit_denominator()).denominator)
                n_num.append((ENcalc(repsum,repsub)[1]).numerator)
                n_den.append((ENcalc(repsum,repsub)[1]).denominator)
                new_mod.append(np.concatenate((np.array(j[0]), -np.array(j[1]))))
        with h5.File('output/data/mass{}/fullNQ{}.h5'.format(int(mass/10**7),q), 'w') as f:
            f.create_dataset("E_numerator", data=e_num, dtype='i4')
            f.create_dataset("E_denominator", data=e_den, dtype='i4')
            f.create_dataset("N_numerator", data=n_num, dtype='i4')
            f.create_dataset("N_denominator", data=n_den, dtype='i4')
#            f.create_dataset("LP", data=LP, dtype='f8')
            f.create_dataset("model", data=np.array(new_mod))
    print('\nMass {} done in {} minutes'.format(mass, (time.time()-start)/60))
        