# constants.py

import numpy as np

from scipy.special import zeta

# The Planck mass M_P = 1.220890(14) x 10^19 GeV
M_PLANCK = 1.220890e19 # GeV
# The reduced Planck mass
M_PLANCK_RED = M_PLANCK/np.sqrt(8*np.pi)
# The Z boson mass M_Z = 91.1880(20) GeV [PDG 2024]
MASS_Z = 91.1880 # GeV
# The strong coupling constant at the Z boson mass
ALPHA_S_MZ = 0.1173
ALPHA_S_SQ = ALPHA_S_MZ*ALPHA_S_MZ
# The fine structure constant
ALPHA_EM = 1.0/137.035999206
# zeta(3) = 1.2020569...
ZETA3 = zeta(3)
# Planck '18 result: omh2 = 0.1198(12) [arXiv:1807.06209]
PLANCK18_OMH2_LIMIT = 0.1222

# Solver-specific constant
Q_DILUTION_FAC = 1e-20