# model_building.py

import os

import matplotlib.pyplot as plt
import numpy as np

from numba import njit #, vectorize
from scipy.integrate import solve_ivp

from .constants import *
from .utils import sign

file_path = os.path.dirname(os.path.realpath(__file__))
repinfo = np.genfromtxt(file_path+"/data/rep_info.dat", dtype='int64')

def print_replist():
    print(repinfo)
    
@njit
def get_max_index(d: int = 6) -> int:
    return np.where(repinfo[:,3] == d)[0][-1]

@njit
def dynkins(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][4:7]

@njit
def casimirs(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    return repinfo[rep_index][7:10]

@njit
def charges_from_rep(rep: int, repinfo: np.ndarray = repinfo) -> tuple[int, ...]:
    ind, sgn = abs(rep) - 1, sign(rep)
    r3, r2, r1 = repinfo[ind][:3]
    return r3, r2, r1, ind, sgn

@njit
def min_dim_from_rep(rep: int, repinfo: np.ndarray = repinfo) -> int:
    ind = abs(rep) - 1
    d_min = repinfo[ind][3]
    return d_min

@njit
def encalc_times_36(reps: np.ndarray[int], repinfo: np.ndarray = repinfo) -> tuple[int, int]:
    e, n = 0, 0
    for rep in reps:
        r3, r2, r1, ind, sgn = charges_from_rep(rep, repinfo)
        d3, _, _ = dynkins(ind, repinfo)
        e += sgn*r3*r2*(3*(r2*r2 - 1) + r1*r1) # Recall that r1 is multiplied by 6
        n += sgn*r2*d3 # Recall that Dynkin labels are multiplied by 36
    return e, n

@njit
def running_Q_contrib(model: np.ndarray[int], repinfo: np.ndarray = repinfo) -> tuple[np.ndarray[float], ...]:
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

@njit
def running(t, y, a_SM, b_SM, a_bSM, b_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    a = a_SM + (t > tQ)*a_bSM
    b = b_SM + (t > tQ)*b_bSM
    # Once y < 0, we are in the unphysical region; however, for easier root finding,
    # we should a bit deeper in this regime by replacing 1/y -> abs(1/y).
    dydt = -a - b.dot(1/np.abs(y))/(4*np.pi)
    return dydt

"""
@njit
def jac(t, y, unused1, b_SM, unused2, b_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    b = b_SM + (t > tQ)*b_bSM
    d2ydij = (b/(y*y))/(4*np.pi)
    return d2ydij

@njit
def start_diverging(unused1: float, y: np.ndarray[float], *unused2: tuple[any, ...]) -> float:
    return min(y) - 0.1
start_diverging.direction = -1
start_diverging.terminal = True
"""

@njit
def hit_LP(unused1: float, y: np.ndarray[float], *unused2: tuple[any, ...]) -> float:
    return min(y)
hit_LP.direction = -1
hit_LP.terminal = True

n_g = 3
a_SM = np.array([4*n_g/3.0 + 0.1, -22/3.0 + 4*n_g/3.0 + 1/6.0, -11 + 4*n_g/3.0])
b1 = np.array([[0, 0, 0], [0, 136/3.0, 0], [0, 0, 102]])
b2 = n_g*np.array([[19/15.0, 0.2, 11/30.0], [0.6, 49/3.0, 1.5], [44/15.0, 4, 76/3.0]])
b3 = np.array([[9/50.0, 0.3, 0], [0.9, 13/6.0, 0], [0, 0, 0]])
b_SM = (-b1+b2+b3).T

@njit
def param_conversion(t: float) -> float:
    return MASS_Z*np.exp(2*np.pi*t)

@njit
def inv_param_conversion(erg: float) -> float:
    return np.log(erg/MASS_Z)/(2*np.pi)

def find_LP(model: list[int], mQ: float = 5e11, lp_threshold: float = 1e18, verbose: bool = True, plot: bool = False) -> tuple[float, int]:
    """
    Find the (lowest) Landau pole (LP) in the running of the gauge couplings.
    
    Parameters
    ----------
    model : list[int]
        List of representations of the heavy quarks
    mQ : float
        Mass of the heavy quarks in GeV
    lp_threshold : float
        Threshold for the LP in GeV (default: 1e18 GeV)
    verbose : bool
        Whether to print additional information or not (default: True)
    plot : bool
        Whether to plot the running of the gauge couplings or not (default: False)
    
    Returns
    -------
    tuple[float, int]
        A tuple of the mass scale of the LP in GeV and the index of the LP in the array of gauge couplings
        
    Notes
    -----
    - The running of the gauge couplings is solved using a (4,5)-Runge-Kutta method with adaptive step size control
    - If no LP is found below the max. scale considered, the value for the LP is set to np.inf
    """
    model_arr = np.array(model, dtype='int')
    a_bSM, b_bSM = running_Q_contrib(model_arr, repinfo)
    t0, t1 = 0, np.log(lp_threshold/MASS_Z)/(2*np.pi) + 10
    # Initial values for \f$\alpha^{-1}\f$ at the Z boson mass MASS_Z ~ 91.2 GeV
    y0 = np.array([1/0.016923, 1/0.03374, 1/0.1173])
    # sol1 = solve_ivp(running, (t0, t1), y0, args=(a_SM, b_SM, a_bSM, b_bSM, mQ), events=start_diverging, method='RK45', rtol=1e-7, atol=1e-7) # , first_step=0.1) # , max_step=1e-2)
    # t0 = sol1.t[-1]
    # y0 = sol1.y[:,-1]
    sol = solve_ivp(running, (t0, t1), y0, args=(a_SM, b_SM, a_bSM, b_bSM, mQ), events=hit_LP, method='RK45', rtol=3e-14, atol=1e-6, first_step=0.1)
    if sol.status == 1:
        tLP = sol.t_events[0][0]
    else:
        mu_stop = param_conversion(sol.t[-1])
        if mu_stop < lp_threshold:
            raise RuntimeError("ERROR. Solver stopped at {:.2e} below the threshold of {:.2e} GeV! {:s}".format(mu_stop, lp_threshold, sol.message))
        elif verbose:
            print(f"INFO. No Landau pole found below {param_conversion(t1):.2e} GeV; setting the LP scale to inf.")
        tLP = np.inf
    indLP = np.argmin(sol.y[:,-1])
    muLP = param_conversion(tLP)
    if plot:
        # mu1 = convert(sol1.t)
        mu2 = param_conversion(sol.t)
        for i,c in enumerate(['r', 'b', 'orange']):
            # plt.plot(mu1, sol1.y[i], c=c, ls='--')
            plt.plot(mu2, sol.y[i], c=c, label=f"$\\alpha_{(i+1):d}$")
        plt.gca().axvline(MASS_Z*np.exp(2*np.pi*tLP), c='k', ls='--')
        plt.gca().axhline(0, c='k', ls='--')
        plt.xscale('log')
        plt.xlabel(r'$\mu$ [GeV]')
        plt.ylabel(r'$\alpha^{-1}$')
        plt.legend()
        plt.savefig("running_SMpre.pdf")
    return muLP, indLP

"""
@njit
def sm_running(t, _, a_SM, a_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    a = a_SM + (t > tQ)*a_bSM
    return [-a]
""";

"""
def running_of_alpha(mQ: float = 5e11):
    t0 = 0
    y0 = [1/0.1173]
    de = (np.log10(mQ)-np.log10(0.15))/200
    ergvals, invalphavals = [], []
    t1 = inv_param_conversion(0.15)
    sol = solve_ivp(simple_running, (t0, t1), y0, args=(a_SM[2], 0, mQ), method='RK45', dense_output=True)
    ergs = np.arange(np.log10(0.15), 0, de)
    ergvals += list(ergs)
    invalphavals += list(sol.sol(inv_param_conversion(10**ergs))[0])
    t1 = inv_param_conversion(mQ)
    sol = solve_ivp(simple_running, (t0, t1), y0, args=(a_SM[2], 0, mQ), method='RK45', dense_output=True)
    ergs = np.arange(0, np.log10(mQ)+de, de)
    ergvals += list(ergs)
    invalphavals += list(sol.sol(inv_param_conversion(10**ergs))[0])
    return ergvals, invalphavals
""";