import h5py as h5
import numpy as np
import sys
import time

from datetime import datetime
from fractions import Fraction
from itertools import combinations_with_replacement
from mpi4py import MPI
from sympy.utilities.iterables import multiset_partitions

def current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

repdict = dict({
     1: [3, 1, Fraction(-1,3)],  2: [3, 1, Fraction(2,3)],    3: [3, 2, Fraction(1,6)],
     4: [3, 2, Fraction(-5,6)],  5: [3, 2, Fraction(7,6)],    6: [3, 3, Fraction(-1,3)],
     7: [3, 3, Fraction(2,3)],   8: [3, 3, Fraction(-4,3)],   9: [6, 1, Fraction(-1,3)],
    10: [6, 1, Fraction(2,3)],  11: [6, 2, Fraction(1,6)],   12: [8, 1, -1],
    13: [8, 2, Fraction(-1,2)], 14: [15, 1, Fraction(-1,3)], 15: [15, 1, Fraction(2,3)],
    16: [3, 3, Fraction(5,3)],  17: [3, 4, Fraction(1,6)],   18: [3, 4, Fraction(-5,6)],
    19: [3, 4, Fraction(7,6)],  20: [15, 2, Fraction(1,6)]
})

nr = len(repdict)

dykindict_su3 = dict({1: 0, 3: Fraction(1,2), 6: Fraction(5,2), 8: 3, 10: Fraction(15,2), 15: 10 })

def encalc(summed, subbed=[]):
    e, n = 0, 0
    for rep in summed:
        e += rep[0]*rep[1]*((rep[1]*rep[1] - 1)*Fraction(1,12) + rep[2]*rep[2])
        n += rep[1]*dykindict_su3[rep[0]]
    for rep in subbed:
        e -= rep[0]*rep[1]*((rep[1]*rep[1] - 1)*Fraction(1,12) + rep[2]*rep[2])
        n -= rep[1]*dykindict_su3[rep[0]]
    return e, n

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ncores = comm.Get_size()
start_time = time.time()

# Initialise variables
output_path = sys.argv[1]
nq = int(sys.argv[2])
start_index = 0
if len(sys.argv) > 3:
   start_index = int(sys.argv[3])

# Compute the number of cases, batch sizes, etc.
models = list(combinations_with_replacement(range(1,nr+1), nq))
n_models = int(len(models) - start_index)
n_batch_size = min(int(0.5*n_models/ncores)+1, 10000)
n_batches = int(n_models/n_batch_size)
if n_batch_size*n_batches < n_models:
    n_batches += 1
five_percent_batch = max(int(0.05*n_batches), 1)

# The main function
def run_process_batch(index):
    evals, nvals = [], []
    i0 = start_index+index*n_batch_size
    i1 = i0 + n_batch_size
    for m in models[i0:i1]:
        repsum = [repdict[k] for k in m]
        e_temp, n_temp = encalc(repsum)
        evals.append(e_temp)
        nvals.append(n_temp)
        for set1,set2 in multiset_partitions(m,2):
            if len(set1) >= len(set2):
                repsum = [repdict[k] for k in set1]
                repsub = [repdict[k] for k in set2]
            else:
                repsum = [repdict[k] for k in set2]
                repsub = [repdict[k] for k in set1]
            e_temp, n_temp = encalc(repsum, repsub)
            evals.append(e_temp)
            nvals.append(n_temp)
    res = []
    for e,n in zip(evals,nvals):
        res.append(e.numerator)
        res.append(e.denominator)
        res.append(n.numerator)
        res.append(n.denominator)
    return np.array(res, dtype='i')

def write_res(dat, rank, index):
    res = dat.reshape((len(dat)//4,4))
    out_file_name = output_path+'/catalogue_NQ_{:d}_i0_{:d}_worker_{:d}_batch_{:d}.hdf5'.format(nq, start_index, rank, index)
    with h5.File(out_file_name, 'w') as f:
        f.create_dataset("e_num", data=res[:,0], dtype='i')
        f.create_dataset("e_den", data=res[:,1], dtype='i')
        f.create_dataset("n_num", data=res[:,2], dtype='i')
        f.create_dataset("n_den", data=res[:,3], dtype='i')
    print('{} | Rank {:d} saved output to file {}.'.format(current_datetime(), rank, out_file_name), flush=True)

# The worker processes compute and save results and receive new tasks.
if (rank > 0):
    t0 = time.time()
    index = rank-1
    res = run_process_batch(index)
    write_res(res, rank, index)
    # comm.Send(res, dest=0, tag=1)
    comm.send(index, dest=0, tag=2)
    for i in range(n_batches):
        task_id = comm.recv(source=0, tag=0)
        if (task_id > n_batches):
            break
        res = run_process_batch(task_id)
        write_res(res, rank, task_id)
        # comm.Send([res, MPI.INT], dest=0, tag=1)
        comm.send(task_id, dest=0, tag=2)
    dt = (time.time()-t0)/60.0
    print(f"{current_datetime()} | MPI rank {rank:d} finished! Calculations took {dt:.1f} mins.")

# The main process distributes tasks.
if (rank == 0):
    print('Main process waiting for {} results from {} other processes...'.format(n_batches, ncores-1), flush=True)
    print('{} | Checks: {} models/batch, {} batches, {} 5pc-batch'.format(current_datetime(), n_batch_size, n_batches, five_percent_batch), flush=True)
    all_results = np.empty(shape=(0,4), dtype='i')
    task_ids = [] 
    for task_id in range(ncores-1, n_batches+ncores):
        info = MPI.Status()
        # comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=info)
        # count = info.Get_elements(MPI.INT)
        # worker_id = info.Get_source()
        # buf = np.empty(count, dtype='i')
        # comm.Recv(buf, source=worker_id, tag=1)
        # buf = buf.reshape((count//4,4))
        # all_results = np.concatenate((all_results,buf))
        # task_ids.append(comm.recv(source=worker_id, tag=2))
        completed_task_id = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=info)
        task_ids.append(completed_task_id)
        worker_id = info.Get_source()
        comm.send(task_id, dest=worker_id, tag=0)
        if (((task_id-ncores+2) % five_percent_batch == 0) | (task_id == n_batches+ncores-1)):
            print('{} | Calculated another batch of size {:d} for task id {}: {:.1f}% complete'.format(current_datetime(), n_batch_size, task_id, float((100.0*task_id)/n_batches)), flush=True)
            np.savetxt(output_path+'/catalogue_NQ_{:d}_{:d}_task_ids.txt'.format(nq, start_index), task_ids, fmt='%d')
    print('{} | All MPI tasks finished after {:.1f} mins!'.format(current_datetime(), (time.time()-start_time)/60.0), flush=True)
    #out_file_name = output_path+'/catalogue_NQ_{:d}_all_partitions_{:d}.hdf5'.format(nq, start_index)
    #print('{} | Formatting results and saving them to '+out_file_name+'.', flush=True)
    #with h5.File(out_file_name, 'w') as f:
    #    f.create_dataset("e_num", data=all_results[:,0], dtype='i')
    #    f.create_dataset("e_den", data=all_results[:,1], dtype='i')
    #    f.create_dataset("n_num", data=all_results[:,2], dtype='i')
    #    f.create_dataset("n_den", data=all_results[:,3], dtype='i')
    #print('{} | All tasks complete! Finishing MPI routine now...', flush=True)