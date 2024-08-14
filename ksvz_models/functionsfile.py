# functionsfile.py

import os

import csv
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from fractions import Fraction
from numba import njit
# from numba.typed import List
from scipy.integrate import solve_ivp

from .constants import *
from .utils import sign

def dynkind(n, r):
    dyn = [defaultdict(lambda : r*r), {1:0,2:1/2,3:2,4:5,5:10,6:35/2,7:28,8:42}, {1:0,3:1/2,6:5/2,8:3,10:15/2,15:10,152:35/2,21:35,24:25,27:27,28:63,35:105/2,36:105,42:119/2,45:165,48:98,55:495/2,60:115}]
    return Fraction(dyn[n-1][r])

def casimir(n, r):
    cas = [defaultdict(lambda : r*r), {1:0,2:3/4,3:2,4:15/4,5:6,6:35/4,7:12,8:63/4}, {1:0,3:4/3,6:10/3,8:3,10:6,15:16/3,152:28/3,21:40/3,24:25/3,27:8,28:126/7,35:12,36:70/3,42:34/3,45:88/3,48:49/3,55:36,60:46/3}]
    return Fraction(cas[n-1][r])

# Read original_list from CSV file
file_path = os.path.dirname(os.path.realpath(__file__))
repinfo = []
with open(file_path+"/data/Q_reps_refined.csv", 'r') as file:
    table = csv.reader(file)
    next(table) # Skip the header row
    for row in table:
        # N.B. q_SU(3) x 6, Dynkin indices x 2, Casimir x 84
        info = [int(q) for q in row[:4]]
        if (info[0] == 8 or info[0] == 27) and (info[2] < 0):
            continue
        info += [int(2*dynkind(3-i, info[i])) for i in range(3)]
        info += [int(84*casimir(3-i, info[i])) for i in range(3)]
        repinfo.append(info)
repinfo = np.array(repinfo, dtype='int64')
repinfo = repinfo[repinfo[:,3].argsort()]

def print_replist():
    print(repinfo)

@njit('int64[:](int64,int64[:,:])')
def dynkins(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][4:7]

@njit('int64[:](int64,int64[:,:])')
def casimirs(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][7:10]

@njit('UniTuple(int64, 5)(int64,int64[:,:])')
def charges_from_rep(rep: int, replist) -> tuple[int, ...]:
    ind, sgn = abs(rep-1), sign(rep)
    r0, r1, r2 = replist[ind][:3]
    return r0, r1, r2, ind, sgn

@njit
def encalc(reps: list[int], repinfo: np.ndarray = repinfo) -> tuple[int, int]:
    e, n = 0, 0
    for rep in reps:
        r0, r1, r2, ind, sgn = charges_from_rep(rep, repinfo)
        d3, _, _ = dynkins(ind, repinfo)
        e += sgn*r0*r1*(3*(r1*r1 - 1) + r2*r2)
        n += 18*sgn*r1*d3
    return e, n

def running_Q_contrib(model: list[int], repinfo: np.ndarray = repinfo) -> tuple[np.ndarray[float], ...]:
    a_bSM = np.zeros(3)
    b_bSM = np.zeros((3,3))
    kappa = 1
    for rep in model:
        r2, r1, r0, ind, _ = charges_from_rep(rep, repinfo)
        d3, d2, d1 = dynkins(ind, repinfo)/2.0
        c3, c2, c1 = casimirs(ind, repinfo)/84.0
        r0 *= np.sqrt(0.6)
        r0 /= 6.0
        a_bSM[0] += 4*kappa*d1*r1*r2/3.0
        a_bSM[1] += 4*kappa*d2*r2/3.0
        a_bSM[2] += 4*kappa*d3*r1/3.0
        b_bSM[0][0] += 4*kappa*d1*c1*r1*r2/3.0
        b_bSM[1][1] += kappa*d2*(4*c2 + 40/3.0)*r2
        b_bSM[2][2] += kappa*d3*(4*c3 + 20)*r1
        b_bSM[0][1] += 4*kappa*c2*d1*r1*r2
        b_bSM[1][0] += 4*kappa*c1*d2*r2
        b_bSM[0][2] += 4*kappa*c3*d1*r1*r2
        b_bSM[2][0] += 4*kappa*c1*d3*r1
        b_bSM[1][2] += 4*kappa*c3*d2*r2
        b_bSM[2][1] += 4*kappa*c2*d3*r1
    return a_bSM, b_bSM

def running(t, y, a_SM, b_SM, a_bSM, b_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    a = a_SM + (t > tQ)*a_bSM
    b = b_SM + (t > tQ)*b_bSM
    dydt = -a - np.matmul(b,1/y)/(4*np.pi)
    return dydt

def hit_LP(_, y, *args):
   return min(1/y)
hit_LP.terminal = True

n_g = 3
a_SM = np.array([4*n_g/3.0 + 0.1, -(22/3.0 - 4*n_g/3.0 - 1/6.0), -(11-4*n_g/3.0)])
b1 = np.array([[0, 0, 0], [0, 136/3.0, 0], [0, 0, 102]])
b2 = n_g*np.array([[19/15.0, 0.2, 11/30.0], [3/5.0, 49/3.0, 1.5], [44/15.0, 4, 76/3.0]])
b3 = np.array([[9/50.0, 3/10.0, 0],[0.9, 13/6.0, 0],[0, 0, 0]])
b_SM = (-b1+b2+b3).T

def find_LP(model: list[int], mQ: float = 5e11, plot: bool = False) -> float:
    a_bSM, b_bSM = running_Q_contrib(model)
    t0, t1 = 0, 20
    y0 = np.array([1/0.016923, 1/0.03374, 1/0.1173]) # \alpha^{-1} ar m_Z = 91.188 GeV
    sol = solve_ivp(running, (t0, t1), y0, args=(a_SM, b_SM, a_bSM, b_bSM, mQ), events=hit_LP)
    try:
        tLP = sol.t_events[0][0]
    except IndexError:
        tLP = t1
    indLP = np.argmin(1/sol.y)
    muLP = MASS_Z*np.exp(2*np.pi*tLP)
    if plot:
        mu = MASS_Z*np.exp(2*np.pi*sol.t)
        g = np.sqrt(4*np.pi/sol.y)
        for i in range(3):
            plt.plot(mu, g[i], label=f"$g_{(i+1):d}$")
        plt.gca().axvline(MASS_Z*np.exp(2*np.pi*tLP))
        plt.xscale('log')
        plt.xlabel(r'$\mu$ [GeV]')
        plt.ylabel(r'$g$')
        plt.legend()
        plt.savefig("running_SMpre.pdf")
    return muLP, indLP