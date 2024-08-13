# cosmo.py

import numpy as np
import os

from numba import njit

from .constants import *
from .utils import fast_factorial

# Cosmological data functions import
file_path = os.path.dirname(os.path.realpath(__file__))
gdata = np.genfromtxt(file_path+"/data/eff_dof_and_aux_functions_1803_01038.dat")
gdata[:,0] -= 9 # Convert from eV to GeV 
gdata[:,3] *= -1

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
def gamma_scaling(te: float) -> float:
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,3])
   else:
      return 1

@njit
def gammad(mQ: float, scale: float = M_PLANCK, d: int = 5) -> float:
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
   nf = d-3
   nfm2fac = fast_factorial(nf-2)
   return 0.25*mQ*pow(mQ/scale, p1)/(pow(4*np.pi, 2*nf-3)*nfm2fac*nfm2fac*(nf-1))

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
      Entropy number density
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
      Energy density
   """
   te2 = te*te
   return c_rho*geff(te)*te2*te2