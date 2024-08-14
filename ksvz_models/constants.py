# constants.py

import numpy as np

# The Planck mass M_P = 1.220890(14) x 10^19 GeV
M_PLANCK = 1.220890e19 # GeV
# The reduced Planck mass
M_PLANCK_RED = M_PLANCK/np.sqrt(8*np.pi)
# The Z boson mass M_Z = 91.1880(20) GeV [PDG 2024]
MASS_Z = 91.1880 # GeV
# The strong coupling constant at the Z boson mass
ALPHA_S = 0.2
ALPHA_S_SQ = ALPHA_S*ALPHA_S