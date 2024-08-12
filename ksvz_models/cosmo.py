# cosmo.py
# Cosmological thermal functions, decay and annhiliation operators

import numpy as np

from numba import njit

from .constants import *
from .utils import fast_factorial

@njit
def gammad(mQ, scale=mP, d=5):
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