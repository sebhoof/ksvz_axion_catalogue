# model_building.py

import os

import matplotlib.pyplot as plt
import numpy as np

from numba import njit
from scipy.integrate import solve_ivp

from .constants import *
from .utils import sign

file_path = os.path.dirname(os.path.realpath(__file__))
repinfo = np.genfromtxt(file_path+"/data/rep_info.dat", dtype='int64')
alphaSinfo = np.load(file_path+"/data/running_alphaS_SM.npy")

def print_replist():
    """
    Print the list of representations from the data file.
    """
    print(repinfo)
    
@njit
def get_max_index(d: int = 6) -> int:
    """
    Return the maximum index of the representations with dimension d (useful for making a subselection).

    Parameters
    ----------
    d : int
        Dimension of the representations (default: 6)

    Returns
    -------
    int
        Maximum index of the representations with dimension d
    """
    return np.where(repinfo[:,3] == d)[0][-1]

@njit
def dynkins(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    """
    Return the Dynkin labels of a representation

    Parameters
    ----------
    rep_index : int
        Index of the representation
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    np.ndarray[int]
        Array of Dynkin labels
    """
    return repinfo[rep_index][4:7]

@njit
def casimirs(rep_index: int, repinfo: np.ndarray = repinfo) -> np.ndarray[int]:
    """
    Return the Casimir invariants of a representation

    Parameters
    ----------
    rep_index : int
        Index of the representation
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    np.ndarray[int]
        Array of Casimir invariants
    """
    return repinfo[rep_index][7:10]

@njit
def charges_from_rep(rep: int, repinfo: np.ndarray = repinfo) -> tuple[int, ...]:
    """
    Return the charges of a representation

    Parameters
    ----------
    rep : int
        Representation index
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    tuple[int, ...]
        Tuple of the charges of the representation, the index, and the "sign" of the representation
    """
    ind, sgn = abs(rep) - 1, sign(rep)
    r3, r2, r1 = repinfo[ind][:3]
    return r3, r2, r1, ind, sgn

@njit
def min_dim_from_rep(rep: int, repinfo: np.ndarray = repinfo) -> int:
    """
    Return the minimum dimension any the decay operator that can be associated with a given representation

    Parameters
    ----------
    rep : int
        Representation index
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    int
        Minimum dimension of any associated decay operator
    """
    ind = abs(rep) - 1
    d_min = repinfo[ind][3]
    return d_min

@njit
def encalc_times_36(reps: np.ndarray[int], repinfo: np.ndarray = repinfo) -> tuple[int, int]:
    """
    Compute the anomaly coefficients E and N for a given set of representations

    Parameters
    ----------
    reps : np.ndarray[int]
        Array of representation indices
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    tuple[int, int]
        Anomaly coefficients \f$36E\f$ and \f$36N\f$

    Notes
    -----
    - The anomaly coefficients are multiplied by 36 to represent them as integers
    """
    e, n = 0, 0
    for rep in reps:
        r3, r2, r1, ind, sgn = charges_from_rep(rep, repinfo)
        d3, _, _ = dynkins(ind, repinfo)
        e += sgn*r3*r2*(3*(r2*r2 - 1) + r1*r1) # Recall that r1 is multiplied by 6
        n += sgn*r2*d3 # Recall that Dynkin labels are multiplied by 36
    return e, n

@njit
def running_Q_contrib(model: np.ndarray[int], repinfo: np.ndarray = repinfo) -> tuple[np.ndarray[float], ...]:
    """
    Compute the the contributions from the heavy quarks (\f$\mathcal{Q}\f$s) to the running of the gauge couplings

    Parameters
    ----------
    model : np.ndarray[int]
        Array of representation indices of the heavy quarks
    repinfo : np.ndarray
        Array of representation information (default: repinfo from file)

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[float]]
        Tuple of the coefficients \f$a_{\mathcal{Q}}\f$ and \f$b_{\mathcal{Q}}\f$ for the heavy quarks
    """
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
    """
    Equation for the running of the gauge couplings at 2-loop order

    Parameters
    ----------
    t : float
        RG integration variable
    y : np.ndarray[float]
        Array of the inverse gauge couplings
    a_SM : np.ndarray[float]
        Array of 1-loop coefficients for the SM contributions
    b_SM : np.ndarray[float]
        Array of 2-loop coefficients for the SM contributions
    a_bSM : np.ndarray[float]
        Array of 1-loop coefficients for the heavy quark contributions
    b_bSM : np.ndarray[float]
        Array of 2-loop coefficients for the heavy quark contributions
    mQ : float
        Mass of the heavy quarks in GeV

    Returns
    -------
    np.ndarray[float]
        Array of the derivatives of the inverse gauge couplings
    """
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    a = a_SM + (t > tQ)*a_bSM
    b = b_SM + (t > tQ)*b_bSM
    # Once y < 0, we are in the unphysical region; however, for easier root finding,
    # we should a bit deeper in this regime by replacing 1/y -> abs(1/y).
    dydt = -a - b.dot(1/np.abs(y))/(4*np.pi)
    return dydt

"""
@TODO Can potentially use the Jacobian for the solver
@njit
def jac(t, y, unused1, b_SM, unused2, b_bSM, mQ):
    tQ = np.log(mQ/MASS_Z)/(2*np.pi)
    b = b_SM + (t > tQ)*b_bSM
    d2ydij = (b/(y*y))/(4*np.pi)
    return d2ydij
"""

@njit
def hit_LP(unused1: float, y: np.ndarray[float], *unused2: tuple[any, ...]) -> float:
    """
    Event function to stop the integration when the Landau pole is hit

    Returns
    -------
    float
        Minimum value of the inverse gauge couplings (zero when the Landau pole is hit)

    Notes
    -----
    - The parameters unused1 and unused2 are only there to match the signature of the event function
    - The event is terminal
    """
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
    """
    Parameter conversion from the RG time to the energy scale \f$E = M_Z\exp(2\pi\mathrm{t})\f$

    Parameters
    ----------
    t : float
        RG variable

    Returns
    -------
    float
        Energy scale in GeV
    """
    return MASS_Z*np.exp(2*np.pi*t)

@njit
def inv_param_conversion(erg: float) -> float:
    """
    Inverse parameter conversion from the energy scale to the RG time \f$\mathrm{t} = \log(E/M_Z)/(2\pi)\f$

    Parameters
    ----------
    erg : float
        Energy scale in GeV

    Returns
    -------
    float
        RG variable
    """
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
    # Initial values for \f$\alpha^{-1}\f$ at the Z boson mass \f$M_Z \approx 91.2\,\f$GeV
    y0 = np.array([1.0/ALPHA_1_MZ, 1.0/ALPHA_2_MZ, 1.0/ALPHA_S_MZ])
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
        mu2 = param_conversion(sol.t)
        for i,c in enumerate(['r', 'b', 'orange']):
            plt.plot(mu2, sol.y[i], c=c, label=f"$\\alpha_{(i+1):d}$")
        plt.gca().axvline(MASS_Z*np.exp(2*np.pi*tLP), c='k', ls='--')
        plt.gca().axhline(0, c='k', ls='--')
        plt.xscale('log')
        plt.xlabel(r'$\mu$ [GeV]')
        plt.ylabel(r'$\alpha^{-1}$')
        plt.legend()
        plt.show()
    return muLP, indLP

@njit
def running_SM(_, y, a_SM, b_SM):
    dydt = -a_SM - b_SM.dot(1/np.abs(y))/(4*np.pi)
    return dydt

def running_of_alphaS(fname: str = "running_alpha_SM.npy") -> np.ndarray[float]:
    """
    Compute the running of the strong coupling constant \f$\alpha_S\f$ in the SM.

    Parameters
    ----------
    fname : str
        File name to save the results (default: "running_alpha_SM.npy")

    Returns
    -------
    np.ndarray[float]
        A 2D array of the energy scale and the inverse of the strong coupling constant \f$\alpha_S^{-1}\f$
    """
    t0, tZ, t1 = inv_param_conversion(0.15), 0, inv_param_conversion(M_PLANCK)
    tvals = np.linspace(t0, t1, 250)
    ergvals, alphavals = [], []
    # Initial values for \f$\alpha^{-1}\f$ at the Z boson mass MASS_Z ~ 91.2 GeV
    yZ = np.array([1.0/ALPHA_1_MZ, 1.0/ALPHA_2_MZ, 1.0/ALPHA_S_MZ])
    # First: running down to the QCD scale
    sol = solve_ivp(running_SM, (tZ, t0), yZ, args=(a_SM, b_SM), method='RK45', rtol=1e-8, atol=1e-8, dense_output=True)
    tvals0 = tvals[tvals <= tZ]
    ergvals += list(param_conversion(tvals0))
    alphavals += list(1/sol.sol(tvals0)[2])
    # Second: running up to the Planck scale
    sol = solve_ivp(running_SM, (tZ, t1), yZ, args=(a_SM, b_SM), method='RK45', rtol=1e-8, atol=1e-8, dense_output=True)
    tvals1 = tvals[tvals > tZ]
    ergvals += list(param_conversion(tvals1))
    alphavals += list(1/sol.sol(tvals1)[2])
    res = np.column_stack((ergvals, alphavals))
    np.save(fname, res, allow_pickle=False)
    return res[res[:,0].argsort()]