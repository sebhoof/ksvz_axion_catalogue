import h5py as h5
import numpy as np
from functionsfile import *
import time
from tqdm import tqdm
import warnings

#masses = 10**7*np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]).astype(float)
#masses = 10**7*np.array([1000000]).astype(float)
masses = [5*10**11];

warnings.filterwarnings("ignore", category=RuntimeWarning)

reps = np.linspace(1,112,112).astype(int)

threshold = 10**18

for mass in masses:
    start = time.time()
    for q in tqdm(range(3,51)):
        e_num, e_den, n_num, n_den, LP = [],[],[],[],[]
        with h5.File('output/data/mass{}/addNQ{}.h5'.format(int(mass/10**7),q-1), 'r') as f:
            prev_lp = f['LP'][:]
            prev_mod = f['model'][:]
        cond = prev_lp >= threshold
        mods_to_ext = prev_mod[cond]
        del(prev_mod)
        del(prev_lp)
        if len(mods_to_ext)==0:
            break
        new_mod = []
        for mod in mods_to_ext:
            for i in range(1, mod[0]+1):
                new_mod.append(np.insert(mod,0,i))
        for mod in tqdm(new_mod):
            temp=[]
            for j in mod:
                temp.append(repdict[j])
            e_num.append((Fraction(ENcalc(temp)[0]).limit_denominator()).numerator)
            e_den.append((Fraction(ENcalc(temp)[0]).limit_denominator()).denominator)
            n_num.append((ENcalc(temp)[1]).numerator)
            n_den.append((ENcalc(temp)[1]).denominator)
            f=[np.array(temp)[:,2], np.array(temp)[:,1], np.array(temp)[:,0]]
            l1,l2,l3=do_it(f, mass)
            LP.append(l1)
        with h5.File('output/data/mass{}/addNQ{}.h5'.format(int(mass/10**7),q), 'w') as f:
            f.create_dataset("E_numerator", data=e_num, dtype='i4')
            f.create_dataset("E_denominator", data=e_den, dtype='i4')
            f.create_dataset("N_numerator", data=n_num, dtype='i4')
            f.create_dataset("N_denominator", data=n_den, dtype='i4')
            f.create_dataset("LP", data=LP, dtype='f8')
            f.create_dataset("model", data=new_mod)
    print('\nMass {} done in {} minutes'.format(mass, (time.time()-start)/60))
        