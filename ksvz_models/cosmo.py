# cosmo.py

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special as sp
import os

from numba import njit
from scipy.integrate import solve_ivp
from sys import path as sysPath

mimes_path = "/Users/sebhoof/Software/mimes/"
sysPath.append(mimes_path+"/src/")
from interfacePy.AxionMass import AxionMass
from interfacePy.Axion import Axion

from .constants import *
from .model_building import alphaSinfo
from .utils import fast_factorial


### Decay operator functions ###

@njit
def gammad(mQ: float, d: int, scale: float = M_PLANCK) -> float:
   """
   Compute a decay rate for a dimension d operator from dimensional analysis.

   Parameters
   ----------
   mQ : float
      Mass of the heavy particle (in GeV)
   scale : float
      Energy scale of the EFT associated with the operator
   d : int
      Dimension of the operator

   Returns
   -------
   float
      Decay rate of the heavy particle (in GeV^{-1})
   """
   if d == 4:
      return 0.125*mQ/np.pi
   p1 = 2*(d-4)
   nfin = d-3
   nfm2fac = fast_factorial(nfin-2)
   return 0.25*mQ*pow(mQ/scale, p1)/(pow(4*np.pi, 2*nfin-3)*nfm2fac*nfm2fac*(nfin-1))


### Annihilation cross section ###

@njit
def sigmav(te, mQ, nf=6, cf=2.0/9, cg=220.0/27, alphaSinfo=alphaSinfo):
   """
   Compute the annihilation cross section for a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   nf : int
      Number of quark flavours that Q can annihilate into (default: 6)
   cf : float
       Group theory factor; 2/9 for triplets, 3/2 for octets (default: 2/9)
   cg : float
      Group theory factor; 220/7 for triplets, 27/4 for octets (default: 220/27)

   Returns
   -------
   float
      Annihilation cross section (in GeV^{-2})
   """
   alph = np.interp(np.log10(te), alphaSinfo[:,0], alphaSinfo[:,1])
   c_ann = nf*cf+cg
   return np.pi*alph*alph*c_ann/(16*mQ*mQ)

### Thermal functions ###

# Cosmological data functions import
file_path = os.path.dirname(os.path.realpath(__file__))
gdata = np.genfromtxt(file_path+"/data/eff_dof_and_transforms_1803_01038.dat")
gdata[:,0] -= 9 # Convert from eV to GeV

@njit
def geff(te: float) -> float:
   """
   Compute the effective number of relativistic degrees of freedom at a given temperature.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Effective number of relativistic degrees of freedom
   """
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,1])
   else:
      return gdata[0,1]

@njit
def geffS(te: float) -> float:
   """
   Compute the effective number of relativistic degrees of freedom at a given temperature for the entropy number density.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Effective number of relativistic degrees of freedom for entropy
   """
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,2])
   else:
      return gdata[0,2]

@njit
def itherm_integrand(xi: float, x: float, pm: int = 1):
   xi2 = xi*xi
   return xi2/(np.exp(np.sqrt(xi2 + x*x)) + pm)

@njit
def jtherm_integrand(xi: float, x: float, pm: int = 1):
   xi2 = xi*xi
   sq = np.sqrt(xi2 + x*x)
   return xi2*sq/(np.exp(sq) + pm)

c_ns = 2.0*np.pi*np.pi/45.0

@njit
def n_s_SM(te: float) -> float:
   """
   Compute the entropy number density of the Standard Model at a given temperature.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Entropy number density (in GeV^3)
   """
   return c_ns*geffS(te)*te*te*te

c_rho = np.pi*np.pi/30.0

@njit
def rho_SM(te: float) -> float:
   """
   Compute the energy density of the Standard Model at a given temperature.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Energy density (in GeV^4)
   """
   te2 = te*te
   return c_rho*geff(te)*te2*te2

@njit
def n_Q_eq_nr(te, mQ, gQ=2):
   """
   The nonrelativistic equilibrium number density of a heavy quark

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   gQ : int
      Number of degrees of freedom for the heavy quark (default: 2)

   Returns
   -------
   float
      Equilibrium number density (in GeV^3)
   """
   x = mQ/te
   return gQ*pow(0.5*x/np.pi,1.5)*np.exp(-x)*pow(te,3)

@njit
def n_Q_eq_r(te, gQ=2):
   """
   Compute the relativistic equilibrium number density of a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   gQ : int
      Number of degrees of freedom for the heavy quark (default: 2)

   Returns
   -------
   float
      Equilibrium number density (in GeV^3)
   """
   return gQ*ZETA3*pow(te,3)/(2*np.pi*np.pi)

@njit
def n_Q_eq_approx(te, mQ, gQ=2):
   """
   Compute the approx. equilibrium number density of a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   gQ : int
      Number of degrees of freedom for the heavy quark (default: 2)

   Returns
   -------
   float
      Equilibrium number density (in GeV^3)
   """
   x = mQ/te
   if x < 1:
      return n_Q_eq_r(te, gQ)
   else:
      return n_Q_eq_nr(te, mQ, gQ)

@njit
def rho_Q_eq_approx(te, mQ, gQ=2):
   """
   Compute the approx. equilibrium energy density of a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   gQ : int
      Number of degrees of freedom for the heavy quark (default: 2)

   Returns
   -------
   float
      Equilibrium energy density (in GeV^4)
   """
   x = mQ/te
   if x < 1:
      return 7*np.pi*np.pi*gQ*pow(te,4)/240
   else:
      return gQ*mQ*pow(te,3)*pow(0.5*x/np.pi,1.5)*np.exp(-x)

@njit
def n_Q_eq(te, mQ, gQ=2):
   """
   Compute the equilibrium number density of a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   gQ : int
      Number of degrees of freedom for the heavy quark (default: 2)

   Returns
   -------
   float
      Equilibrium number density (in GeV^3)
   """
   x = mQ/te
   if x > 1e2:
      return n_Q_eq_nr(te, mQ, gQ)
   elif x < 1e-3:
      return n_Q_eq_r(te, gQ)
   return pow(te,3)*gQ*x*x*sp.kn(2,x)/(2*np.pi*np.pi)

@njit
def e_Q(te: float, mQ: float) -> float:
   """
   Compute the (approximate) energy of a heavy quark in a plasma

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)

   Returns
   -------
   float
      Energy of the heavy quark (in GeV)

   Notes
   -----
   - This is the average energy per \f$\mathcal{Q}\f$ particle, valid for energies not much larger than the mass
   """
   return mQ + 1.5*te


### Evolution of the Universe ###

@njit
def gS_scaling(te: float) -> float:
   """
   The scaling factor related to the effective number of relativistic degrees of freedom in entropy.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Scaling factor for the Boltzmann equation
   """
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,3])
   return 1

@njit
def g_scaling(te: float) -> float:
   """
   The scaling factor related to the effective number of relativistic degrees of freedom in energy density.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Scaling factor for the EOS computation
   """
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,4])
   return 4

@njit
def hubble(te, rhoQ):
   """
   Compute the Hubble parameter.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   rhoQ : float
      Energy density of the heavy quark (in GeV^4)

   Returns
   -------
   float
      Hubble parameter (in GeV)
   """
   rhoR = rho_SM(te)
   return np.sqrt((rhoQ+rhoR)/3)/M_PLANCK_RED


### Boltzmann equation and supporting functions ###

@njit
def crossBBN(_: float, y: np.ndarray[float], *unused) -> float:
   """
   Compute the cross point with the BBN temperature

   Parameters
   ----------
   _ : float
      Integration variable u (not used, for compatibility)
   y : np.ndarray[float]
      Array of equation values
   unused : tuple
      Additional arguments (not used, for compatibility)

   Returns
   -------
   float
      Value of the equation (equals 0 at the cross point)

   Notes
   -----
   - This function is to be used as an "event" for the ODE solver
   """
   return y[0] - 1e-3
crossBBN.terminal = True

@njit
def dilutedQ(_: float, y: np.ndarray[float], mQ: float, *unused) -> float:
   """
   Compute the diluted heavy quark number density

   Parameters
   ----------
   _ : float
      Integration variable u (not used, for compatibility)
   y : np.ndarray[float]
      Array of equation values
   mQ : float
      Mass of the heavy quark (in GeV)
   unused : tuple
      Additional arguments (not used, for compatibility)

   Returns
   -------
   float
      Value of the equation (equals 0 at the diluted point)

   Notes
   -----
   - This function is to be used as an "event" for the ODE solver
   """
   if len(y) == 1:
      return 1
   te, nQ = y[0], y[1:]
   eQ = e_Q(te, mQ)
   rhoQ = nQ*eQ
   rhoSM = rho_SM(te)
   res = min(rhoQ - Q_DILUTION_FAC*rhoSM)
   return res
dilutedQ.terminal = True

@njit
def decayingQs(_: float, y: np.ndarray, mQ: float, qdims: np.ndarray[int], qmult: np.ndarray[int], eft_scale: float = M_PLANCK) -> np.ndarray:
   """
   Compute the RHS of the Boltzmann equation for the heavy quarks

   Parameters
   ----------
   _ : float
      Integration variable u (not used, for compatibility)
   y : np.ndarray[float]
      Array of equation values (first element is the temperature, the remaining are the number densities of the heavy quarks)
   mQ : float
      Mass of the heavy quarks (in GeV)
   qdims : np.ndarray[int]
      Dimensions of the operators
   qmult : np.ndarray[int]
      Multiplicities of the operators
   eft_scale : float
      Energy scale associated with the decay operators (default: M_PLANCK)

   Returns
   -------
   np.ndarray[float]
      Array of the derivatives of the equation values
   """
   te, nQ = y[0], y[1:]
   if len(y) == 1:
      return np.array([-te/gS_scaling(te)])
   nQ_sq = nQ*nQ
   nQeq = qmult*n_Q_eq(te, mQ)
   nQeq_sq = nQeq*nQeq
   eQ = e_Q(te, mQ)
   rhoQ = nQ*eQ
   h = hubble(te, sum(rhoQ))
   s = n_s_SM(te)
   # Include inverse decays
   decays = np.array([gammad(mQ, d=d, scale=eft_scale) for d in qdims])*(nQ - nQeq)/h
   # decays = np.array([gammad(mQ, d=d) for d in qdims])*nQ/h
   annihilations = sigmav(te, mQ)*(nQ_sq - nQeq_sq)/h
   te_eq = [(-te + sum(decays)*eQ/(3*s))/gS_scaling(te)]
   q_eq = -3*nQ - decays - annihilations
   q_eq = list(q_eq)
   return np.array(te_eq + q_eq)

# @TODO: remove
# @njit
# def eos_old(te: float, nQ: float, mQ: float) -> float:
#    """
#    Compute the equation of state parameter w of the Universe from the Boltzmann equation.

#    Parameters
#    ----------
#    te : float
#       Temperature (in GeV)
#    nQ : float
#       Number density of the heavy quark (in GeV^3)
#    mQ : float
#       Mass of the heavy quark (in GeV)

#    Returns
#    -------
#    float
#       Equation of state parameter
#    """
#    nQeq = n_Q_eq(te, mQ)
#    rhoQ = nQ*e_Q(te, mQ)
#    h = hubble(te, rhoQ)
#    rhoR = rho_SM(te)
#    annihilations = sigmav(te, mQ)*(nQ*nQ - nQeq*nQeq)/h
#    dhduh = (3*rhoQ + 4*rhoR + annihilations)/(6*pow(M_PLANCK_RED*h,2))
#    return 2*dhduh/3 - 1

@njit
def eos_SM(te: float) -> float:
   """
   Compute the equation of state parameter w of the Universe for the SM

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Equation of state parameter of the SM
   """
   gamma_ratio = g_scaling(te)/gS_scaling(te)
   return gamma_ratio/3 - 1

EOS_SM_BBN = eos_SM(T_BBN)

@njit
def eos(yvals: np.ndarray[float], qdims: np.ndarray[int], qmult: np.ndarray[int], mQ: float) -> float:
   """
   Compute the equation of state parameter w of the Universe from the Boltzmann equation.

   Parameters
   ----------
   te : float
      Solution
   mQ : float
      Mass of the heavy quark (in GeV)

   Returns
   -------
   float
      Equation of state parameter
   """
   te, nQ = yvals[0], sum(yvals[1:])
   if nQ > 0:
      dydu = decayingQs(None, yvals, mQ, qdims, qmult)
      dTdu, dndu = dydu[0], sum(dydu[1:])
      eQ = e_Q(te, mQ)
      rhoR = rho_SM(te)
      hterm = 3*M_PLANCK_RED*hubble(te, nQ*eQ)
      dh2du = (g_scaling(te)*dTdu*rhoR/te + dndu*eQ + 1.5*dTdu*nQ)
      return - dh2du/(hterm*hterm) - 1
   else:
      return eos_SM(te)

@njit
def dim_signature(qdims: np.ndarray[int], qmult: np.ndarray[int]) -> str:
   mult_string = ""
   for d0 in range(5,9):
      if d0 in qdims:
         mult_string += "_"+str(qmult[qdims==d0][0])
      else:
         mult_string += "_0"
   return mult_string

def compute_cosmology(qdims: np.ndarray[int], qmult: np.ndarray[int], mQ: float, eft_scale: float = M_PLANCK, verbose: bool = False) -> tuple[np.ndarray[float], np.ndarray[float], list[scipy.integrate._ivp.ivp.OdeResult], str]:
   if len(qdims) != len(qmult):
      raise ValueError("The dimensions 'qdims' and multiplicities 'qmult' of the operators must have the same length")
   ubreaks, tebreaks, sols, dims, mults = [], [], [], [], []
   cont, no_bbn = True, True
   te_ini = 10*mQ
   u1 = 0
   ufin1 = np.log(1e21*(mQ/1e11))
   nQ_ini = n_Q_eq(te_ini, mQ)
   y1 = np.array([te_ini] + [m*nQ_ini for m in qmult])
   while cont:
      sol = solve_ivp(decayingQs, [u1, ufin1], y1, args=(mQ, qdims, qmult, eft_scale), dense_output=True, events=(crossBBN, dilutedQ), method='RK45', rtol=1e-7, atol=0)
      sols.append(sol)
      dims.append(qdims)
      mults.append(qmult)
      try:
         u2 = sol.t_events[0][0]
         # If the BBN event is triggered, we stop the solver
         if verbose:
            print(f"Reached BBN at u = {u2:.2f}!", flush=True)
         cont = False
         no_bbn = False
      except IndexError:
         u2 = sol.t[-1]
      try:
         uBreak = sol.t_events[1][0]
         # If a dilution event is triggered, we save the result and continue solving the equation
         u2 = uBreak
         u1 = uBreak
         te = sol.y[0][-1]
         tebreaks.append(te)
         nQ = sol.y[1:][:,-1]
         rhoQ = nQ*e_Q(te, mQ)
         rhoSM = rho_SM(te)
         sel0 = [r > 1.01*Q_DILUTION_FAC*rhoSM for r in rhoQ]
         indices = np.where(sel0)
         if verbose:
            print(f"One type of Q has been diluted at u = {u2:.2f} (T = {te:.2e} GeV): ", sel0, flush=True)
         y1 = np.array([te] + list(nQ[indices]))
         qdims = qdims[indices]
         qmult = qmult[indices]
      except IndexError:
         if no_bbn:
            raise RuntimeError("Stopping solver at u = {:.3f} (T = {:.2e} GeV); this should not happen...".format(sol.t[-1], sol.sol(sol.t[-1])[0]))
      ubreaks.append(u2)
   return ubreaks, tebreaks, sols, dims, mults

def save_cosmology(ubreaks, tebreaks, sols, dims, mults, mQ, output_file, plot=False):
   u2 = ubreaks[-1]
   uvals = np.linspace(0, u2, 500)
   tevals, lnhvals, wvals = [], [], []
   uind = 0
   for u in uvals:
      if u > ubreaks[uind]:
         uind += 1
      s = sols[uind].sol(u)
      te = s[0]
      nQsum = sum(s[1:])
      tevals.append(te)
      # wvals.append(eos_old(te,nQsum,mQ))
      wvals.append(eos(s, dims[uind], mults[uind], mQ))
      eQ = e_Q(te, mQ)
      lnhvals.append(np.log(hubble(te,nQsum*eQ)))
   res = np.array([uvals, tevals, lnhvals]).T
   w_bbn = wvals[-1]
   # N.B. Do not add a header to the output file; MiMeS cannot handle it
   np.savetxt(output_file, res, fmt="%.9f", delimiter="\t")
   if plot:
      plt.plot(1e-3/np.array(tevals), wvals, 'k-')
      plt.gca().axvline(1, c='r', ls='-', label=r"$T_\mathrm{BBN}$")
      plt.gca().axvline(1/150, c='r', ls='--', label=r"$T_\mathrm{QCD}$")
      for tbr in tebreaks:
         plt.gca().axvline(1e-3/tbr, c='grey', ls=':')
      # plt.legend(frameon=False, loc=(0, 0), title=r"$[d_1,\,d_2]$, $m_\mathcal{Q} = 10^{12}\,\mathrm{GeV}$", ncol=1)
      plt.legend(frameon=False)
      plt.xlabel(r"$T_\mathrm{BBN}/T$")
      plt.ylabel(r"Equation of state $w$")
      plt.xscale('log')
      plt.xlim([1e-3/(10*mQ), 2])
      plt.ylim([0, 0.34])
      plt.show()
   return w_bbn, res

def omh2_axion(mQ: float, cosmology: str, thetai: float = 2.2):
   ma = AxionMass(mimes_path+"src/data/chi.dat", 0, M_PLANCK)
   te_max, chi_max, chi_min = ma.getTMax(), ma.getChiMax(), ma.getChiMin()
   @njit
   def ma_high(te, fa):
      return chi_max*pow(te_max/te, 8.16)/(fa*fa)
   @njit
   def ma_low(_, fa):
      return chi_min/(fa*fa)
   ma.set_ma2_MAX(ma_high)
   ma.set_ma2_MIN(ma_low)
   ax = Axion(thetai, mQ, 500, 1e-4, 1e3, 15, 1e-3, cosmology, ma, 1e-1, 1e-8, 1e-1, 1e-11, 1e-11, 0.9, 1.2, 0.8, int(1e7))
   ax.solveAxion()
   omh2 = ax.relic
   # Cleaning up memory is required according to MiMeS documentation
   del ax
   del ma
   return omh2

def omh2_axion_sm(mQ: float, thetai: float = 2.2):
   rd_cosmo = mimes_path+"UserSpace/InputExamples/RDinput.dat"
   return omh2_axion(mQ, rd_cosmo, thetai)