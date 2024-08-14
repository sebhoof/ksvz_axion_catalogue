# functionsfile.py

import os

import matplotlib.pyplot as plt
import numpy as np

from numba import njit
# from numba.typed import List
from scipy.integrate import solve_ivp

from .constants import *
from .utils import sign

file_path = os.path.dirname(os.path.realpath(__file__))
repinfo = np.genfromtxt(file_path+"/data/rep_info.dat", dtype='int64')

def print_replist():
    print(repinfo)
    
def get_max_index(d: int = 6) -> int:
    return np.where(repinfo[:,3] == d)[0][-1]

@njit('int64[:](int64,int64[:,:])')
def dynkins(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][4:7]

@njit('int64[:](int64,int64[:,:])')
def casimirs(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][7:10]

@njit('UniTuple(int64, 5)(int64,int64[:,:])')
def charges_from_rep(rep: int, replist) -> tuple[int, ...]:
    ind, sgn = abs(rep-1), sign(rep)
    r3, r2, r1 = replist[ind][:3]
    return r3, r2, r1, ind, sgn

@njit
def encalc_times_36(reps: list[int], repinfo: np.ndarray = repinfo) -> tuple[int, int]:
    e, n = 0, 0
    for rep in reps:
        r3, r2, r1, ind, sgn = charges_from_rep(rep, repinfo)
        d3, _, _ = dynkins(ind, repinfo)
        e += sgn*r3*r2*(3*(r2*r2 - 1) + r1*r1) # Recall that r1 is multiplied by 6
        n += sgn*r2*d3 # Recall that Dynkin labels are multiplied by 36
    return e, n

def running_Q_contrib(model: list[int], repinfo: np.ndarray = repinfo) -> tuple[np.ndarray[float], ...]:
    a_bSM = np.zeros(3)
    b_bSM = np.zeros((3,3))
    kappa = 1
    for rep in model:
        r3, r2, _, ind, _ = charges_from_rep(rep, repinfo)
        d3, d2, d1 = dynkins(ind, repinfo)/36.0
        c3, c2, c1 = casimirs(ind, repinfo)/36.0
        d1 *= 0.6
        c1 *= 0.6
        a_bSM[0] += d1*r2*r3
        a_bSM[1] += d2*r3
        a_bSM[2] += d3*r2
        b_bSM[0][0] += d1*4*c1*r2*r3
        b_bSM[1][1] += d2*(4*c2 + 40/3.0)*r3
        b_bSM[2][2] += d3*(4*c3 + 20)*r2
        b_bSM[0][1] += 4*c2*d1*r2*r3
        b_bSM[1][0] += 4*c1*d2*r3
        b_bSM[0][2] += 4*c3*d1*r2*r3
        b_bSM[2][0] += 4*c1*d3*r2
        b_bSM[1][2] += 4*c3*d2*r3
        b_bSM[2][1] += 4*c2*d3*r2
    a_bSM *= 4*kappa/3.0
    b_bSM *= kappa
    return a_bSM, b_bSM

def running(t, y, a_SM, b_SM, a_bSM, b_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    a = a_SM + (t > tQ)*a_bSM
    b = b_SM + (t > tQ)*b_bSM
    dydt = -a - np.matmul(b,1/y)/(4*np.pi)
    return dydt

def hit_LP(_, y, *args):
   return min(y)
hit_LP.terminal = True

n_g = 3
a_SM = np.array([4*n_g/3.0 + 0.1, -22/3.0 + 4*n_g/3.0 + 1/6.0, -11 + 4*n_g/3.0])
b1 = np.array([[0, 0, 0], [0, 136/3.0, 0], [0, 0, 102]])
b2 = n_g*np.array([[19/15.0, 0.2, 11/30.0], [0.6, 49/3.0, 1.5], [44/15.0, 4, 76/3.0]])
b3 = np.array([[9/50.0, 0.3, 0], [0.9, 13/6.0, 0], [0, 0, 0]])
b_SM = (-b1+b2+b3).T

def find_LP(model: list[int], mQ: float = 5e11, plot: bool = False) -> float:
    a_bSM, b_bSM = running_Q_contrib(model)
    t0, t1 = 0, 20
    y0 = np.array([1/0.016923, 1/0.03374, 1/0.1173]) # \alpha^{-1} ar m_Z = 91.188 GeV
    sol = solve_ivp(running, (t0, t1), y0, args=(a_SM, b_SM, a_bSM, b_bSM, mQ), events=hit_LP)
    try:
        tLP = sol.t_events[0][0]
    except IndexError:
        print(f"No Landau pole found before t1 < {t1:f}.")
        tLP = t1
    indLP = np.argmin(sol.y[-1])
    muLP = MASS_Z*np.exp(2*np.pi*tLP)
    if plot:
        mu = MASS_Z*np.exp(2*np.pi*sol.t)
        g = np.sqrt(4*np.pi/sol.y)
        for i in range(3):
            plt.plot(mu, g[i], label=f"$g_{(i+1):d}$")
        plt.gca().axvline(MASS_Z*np.exp(2*np.pi*tLP), c='k', ls='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\mu$ [GeV]')
        plt.ylabel(r'$g$')
        plt.legend()
        plt.savefig("running_SMpre.pdf")
    return muLP, indLP