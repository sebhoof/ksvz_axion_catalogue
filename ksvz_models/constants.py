# constants.py

import numpy as np

from scipy.special import zeta

# The Planck mass M_P = 1.220890(14) x 10^19 GeV
M_PLANCK = 1.220890e19 # GeV
# The reduced Planck mass
M_PLANCK_RED = M_PLANCK/np.sqrt(8*np.pi)
# The Z boson mass M_Z = 91.1880(20) GeV [PDG 2024]
MASS_Z = 91.1880 # GeV
# The coupling constant at the Z boson mass [arXiv:1208.3357], Eq. (31)
ALPHA_1_MZ = 0.01692 # +/- 0.00004
ALPHA_2_MZ = 0.033735 # +/- 0.000020
ALPHA_S_MZ = 0.1173 # +/- 0.0007
ALPHA_S_SQ = ALPHA_S_MZ*ALPHA_S_MZ

# The fine structure constant [PDG 2024]
ALPHA_EM = 1.0/137.035999206
# zeta(3) = 1.2020569...
ZETA3 = zeta(3)
# Planck '18 result: omh2 = 0.1198(12) [arXiv:1807.06209]
PLANCK18_OMH2_LIMIT = 0.1222

# Solver-specific constant
Q_DILUTION_FAC = 1e-20
# BBN temperature
T_BBN = 0.001 # GeV
# QCD temperature, e.g., [arXiv:1812.08235]
T_QCD = 0.160 # GeV