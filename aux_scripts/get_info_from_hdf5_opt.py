import sys
import time
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from fractions import Fraction
from tqdm import tqdm

start = time.time()

filename = sys.argv[1]+'.hdf5'
with h5.File(filename, 'r') as f:
    e_num = f['e_num'][:]
    e_den = f['e_den'][:]
    n_num = f['n_num'][:]
    n_den = f['n_den'][:]

time1 = time.time()
print('Read in HDF5 file {} after {:.2f} mins.'.format(filename, (time1-start)/60.0), flush=True)

n_total = len(n_num)

#cond = (n_num != 0)
#n_vals = np.array([Fraction(nn,nd) for nn,nd in zip(n_num[cond],n_den[cond])])
#n_nnot0 = len(n_vals)
#n_n0 = n_total - n_nnot0

time2 = time.time()
#print('Processed N data after {:.2f} mins.'.format((time2-time1)/60.0), flush=True)

#e_n_ratios = np.array([Fraction(en,ed)/n_fr for en,ed,n_fr in zip(e_num[cond],e_den[cond],n_vals)])

e_n_ratios = []
e_n_ratio_counts = []
e_n_hist = []
counts = []

##counter = 0
##for i,nn,nd,en,ed in enumerate(zip(n_num, n_den, e_num, e_den)):
##   if (100*i/n_total) >= (5*counter):
##      print('{:d}%% finished ({:d} out of {:d})'.format(5*counter, i, n_total), flush = True)
##      counter += 1 
##   el = [nn,nd,en,ed]
##   if el in e_n_hist:
##      i = e_n_hist.index(el)
##      counts[i] += 1
##   else:
##      e_n_hist.append(el)
##      counts.append(1)

##counts = np.array(counts, dtype='i')

e_n_hist, counts = np.unique(np.array((n_num, n_den, e_num, e_den)).T, axis=0, return_counts=True)

time3 = time.time()
print('Processed E and N data after {:.2f} mins.'.format((time3-time2)/60.0), flush=True)

n_ndw1 = 0

for el,c in zip(e_n_hist,counts):
   nn,nd,en,ed = el[0],el[1],el[2],el[3]
   if (nn != 0):
      n_fr = Fraction(nn,nd)
      if 2*n_fr == 1:
         n_ndw1 += c
      fr = Fraction(en,ed)/n_fr
      if fr in e_n_ratios:
         i = e_n_ratios.index(fr)
         e_n_ratio_counts[i] += c
      else:
         e_n_ratios.append(fr)
         e_n_ratio_counts.append(c)

e_n_ratios = np.array(e_n_ratios)
e_n_ratio_counts = np.array(e_n_ratio_counts, dtype='i')
      
n_nnot0 = sum(e_n_ratio_counts)
n_n0 = n_total - n_nnot0

time4 = time.time()
print('Processed E/N data after {:.2f} mins.'.format((time4-time3)/60.0), flush=True)

c_vals = np.abs(e_n_ratios.astype('float') - 1.92)
n_pph = sum(e_n_ratio_counts[c_vals < 0.04])

#n_ndw1 = sum(2*n_vals == 1)

time5 = time.time()
print('Got n_pph and n_dw1 after {:.2f} mins.'.format((time5-time4)/60.0), flush=True)

# a, b = np.unique(e_n_ratios, return_counts=True)
n_distinct = len(e_n_ratios)
n_unique = sum(e_n_ratio_counts==1)
max_e_n = e_n_ratios[c_vals == c_vals.max()][0]

outfile = sys.argv[1]+'_new_opt.txt'
np.savetxt(outfile, np.hstack((np.array(e_n_hist, dtype='i'), np.array([counts], dtype='i').T)), fmt='%d')

print('For plot: n_total, n_pph, n_ndw1, n_unique, n_n0, n_distinct', flush=True)
print('{:d} {:d} {:d} {:d} {:d} {:d}'.format(n_total, n_pph, n_ndw1, n_unique, n_n0, n_distinct), flush=True)
print('\nFor table: nN =/= 0: {:.2f} % ({:d})'.format(100.0*n_nnot0/n_total, n_nnot0), flush=True)
print('Max. E/N: {}'.format(e_n_ratios.max()), flush=True)
print('Max. coupling found for E/N = {}'.format(max_e_n), flush=True)

print('\nTotal time taken: {:.2f} min'.format((time.time()-start)/60.0), flush=True)
