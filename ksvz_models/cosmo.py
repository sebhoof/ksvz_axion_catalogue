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
from .utils import fast_factorial


### Decay operator functions ###

@njit
def gammad(mQ: float, d: int = 5, scale: float = M_PLANCK) -> float:
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


### Annihilation cross section ###

# TODO: Compute these at two-loop level
alph0_ergs = np.array([-0.8239087409443188, -0.7612943472179171, -0.6986799534915155, -0.6360655597651138, -0.5734511660387122, -0.5108367723123105, -0.44822237858590885, -0.3856079848595072, -0.32299359113310555, -0.2603791974067039, -0.19776480368030225, -0.1351504099539006, -0.07253601622749895, -0.009921622501097294, 0.0, 0.0626143937264017, 0.1252287874528034, 0.18784318117920507, 0.2504575749056068, 0.3130719686320085, 0.37568636235841013, 0.43830075608481184, 0.5009151498112135, 0.5635295435376152, 0.626143937264017, 0.6887583309904186, 0.7513727247168203, 0.813987118443222, 0.8766015121696237, 0.9392159058960254, 1.001830299622427, 1.0644446933488287, 1.1270590870752304, 1.1896734808016323, 1.252287874528034, 1.3149022682544356, 1.3775166619808372, 1.4401310557072389, 1.5027454494336405, 1.5653598431600424, 1.627974236886444, 1.6905886306128457, 1.7532030243392474, 1.815817418065649, 1.8784318117920509, 1.9410462055184525, 2.003660599244854, 2.066274992971256, 2.1288893866976575, 2.191503780424059, 2.254118174150461, 2.3167325678768624, 2.3793469616032645, 2.441961355329666, 2.504575749056068, 2.5671901427824695, 2.629804536508871, 2.692418930235273, 2.7550333239616744, 2.817647717688076, 2.8802621114144777, 2.9428765051408794, 3.005490898867281, 3.068105292593683, 3.130719686320085, 3.1933340800464864, 3.255948473772888, 3.3185628674992897, 3.3811772612256914, 3.443791654952093, 3.5064060486784947, 3.5690204424048964, 3.631634836131298, 3.6942492298577, 3.7568636235841018, 3.8194780173105034, 3.882092411036905, 3.9447068047633067, 4.007321198489708, 4.06993559221611, 4.132549985942512, 4.195164379668913, 4.257778773395315, 4.320393167121717, 4.383007560848118, 4.44562195457452, 4.508236348300922, 4.570850742027323, 4.633465135753725, 4.696079529480127, 4.758693923206529, 4.821308316932931, 4.883922710659332, 4.946537104385734, 5.009151498112136, 5.071765891838537, 5.134380285564939, 5.196994679291341, 5.259609073017742, 5.322223466744144, 5.384837860470546, 5.447452254196947, 5.510066647923349, 5.5726810416497505, 5.635295435376152, 5.697909829102554, 5.7605242228289555, 5.823138616555357, 5.885753010281759, 5.94836740400816, 6.010981797734562, 6.073596191460965, 6.136210585187366, 6.198824978913768, 6.26143937264017, 6.324053766366571, 6.386668160092973, 6.4492825538193745, 6.511896947545776, 6.574511341272178, 6.6371257349985795, 6.699740128724981, 6.762354522451383, 6.8249689161777845, 6.887583309904186, 6.950197703630588, 7.012812097356989, 7.075426491083391, 7.138040884809793, 7.200655278536194, 7.263269672262596, 7.325884065988998, 7.3884984597154, 7.451112853441802, 7.5137272471682035, 7.576341640894605, 7.638956034621007, 7.7015704283474085, 7.76418482207381, 7.826799215800212, 7.889413609526613, 7.952028003253015, 8.014642396979417, 8.07725679070582, 8.13987118443222, 8.202485578158623, 8.265099971885023, 8.327714365611426, 8.390328759337827, 8.45294315306423, 8.51555754679063, 8.578171940517032, 8.640786334243433, 8.703400727969836, 8.766015121696237, 8.828629515422639, 8.89124390914904, 8.953858302875442, 9.016472696601843, 9.079087090328246, 9.141701484054646, 9.204315877781049, 9.26693027150745, 9.329544665233852, 9.392159058960255, 9.454773452686656, 9.517387846413058, 9.580002240139459, 9.642616633865861, 9.705231027592262, 9.767845421318665, 9.830459815045065, 9.893074208771468, 9.955688602497869, 10.018302996224271, 10.080917389950672, 10.143531783677075, 10.206146177403475, 10.268760571129878, 10.331374964856279, 10.393989358582681, 10.456603752309082, 10.519218146035485, 10.581832539761885, 10.644446933488288, 10.70706132721469, 10.769675720941091, 10.832290114667494, 10.894904508393894, 10.957518902120297, 11.020133295846698, 11.0827476895731, 11.145362083299501, 11.207976477025904, 11.270590870752304, 11.333205264478707, 11.395819658205108, 11.45843405193151, 11.521048445657911, 11.583662839384314, 11.646277233110714, 11.708891626837117])
alph0_vals = np.array([1.3838186599959634, 1.544441773571128, 1.7050648871462917, 1.8656880007214545, 2.026311114296619, 2.1869342278717827, 2.3475573414469455, 2.50818045502211, 2.6688035685972737, 2.8294266821724365, 2.990049795747602, 3.1506729093227657, 3.3112960228979276, 3.471919136473092, 3.4973708226265083, 3.6579939362017324, 3.818617049776954, 3.9792401633521672, 4.139863276927376, 4.300486390502582, 4.461109504077782, 4.62173261765298, 4.782355731228172, 4.942978844803362, 5.103601958378549, 5.2642250719537325, 5.424848185528914, 5.585471299104093, 5.74609441267927, 5.906717526254445, 6.0673406398296175, 6.22796375340479, 6.388586866979961, 6.54920998055513, 6.709833094130298, 6.870456207705465, 7.031079321280631, 7.191702434855797, 7.3523255484309615, 7.512948662006127, 7.6735717755812916, 7.834194889156455, 7.99481800273162, 8.155441116306784, 8.316064229881947, 8.476687343457112, 8.637310457032275, 8.797933570607439, 8.958556684182602, 9.119179797757766, 9.27980291133293, 9.440426024908094, 9.601049138483258, 9.761672252058421, 9.922295365633586, 10.08291847920875, 10.243541592783913, 10.404164706359076, 10.56478781993424, 10.725410933509405, 10.886034047084568, 11.04665716065973, 11.207280274234895, 11.36790338781006, 11.528526501385223, 11.689149614960387, 11.849772728535552, 12.010395842110714, 12.171018955685879, 12.331642069261044, 12.492265182836206, 12.652888296411371, 12.813511409986535, 12.974134523561698, 13.134757637136863, 13.295380750712026, 13.45600386428719, 13.616626977862355, 13.777250091437516, 13.93787320501268, 14.098496318587845, 14.259119432163008, 14.41974254573817, 14.580365659313337, 14.7409887728885, 14.901611886463662, 15.062235000038825, 15.22285811361399, 15.383481227189154, 15.54410434076432, 15.704727454339483, 15.865350567914648, 16.025973681489813, 16.186596795064975, 16.347219908640138, 16.507843022215305, 16.668466135790467, 16.82908924936563, 16.989712362940793, 17.150335476515956, 17.310958590091122, 17.47158170366628, 17.632204817241448, 17.792827930816614, 17.953451044391773, 18.11407415796694, 18.274697271542102, 18.435320385117265, 18.59594349869243, 18.756566612267594, 18.91718972584276, 19.077812839417923, 19.238435952993086, 19.399059066568253, 19.559682180143415, 19.720305293718578, 19.880928407293744, 20.041551520868904, 20.20217463444407, 20.362797748019233, 20.523420861594396, 20.68404397516956, 20.844667088744725, 21.005290202319888, 21.16591331589505, 21.326536429470217, 21.48715954304538, 21.647782656620542, 21.80840577019571, 21.969028883770868, 22.12965199734603, 22.2902751109212, 22.450898224496363, 22.611521338071526, 22.772144451646692, 22.93276756522185, 23.09339067879702, 23.254013792372184, 23.414636905947344, 23.57526001952251, 23.735883133097676, 23.896506246672836, 24.057129360248002, 24.217752473823168, 24.378375587398327, 24.538998700973494, 24.699621814548657, 24.86024492812382, 25.020868041698982, 25.181491155274152, 25.342114268849308, 25.502737382424478, 25.66336049599964, 25.823983609574803, 25.984606723149966, 26.14522983672513, 26.30585295030029, 26.466476063875458, 26.627099177450624, 26.787722291025787, 26.948345404600946, 27.108968518176116, 27.269591631751275, 27.43021474532644, 27.590837858901608, 27.75146097247677, 27.912084086051934, 28.072707199627096, 28.233330313202263, 28.393953426777422, 28.554576540352592, 28.715199653927748, 28.87582276750291, 29.03644588107808, 29.197068994653243, 29.357692108228406, 29.518315221803572, 29.67893833537873, 29.839561448953894, 30.000184562529064, 30.160807676104227, 30.32143078967939, 30.482053903254553, 30.642677016829715, 30.803300130404878, 30.963923243980048, 31.12454635755521, 31.285169471130374, 31.445792584705536, 31.6064156982807, 31.767038811855862, 31.927661925431032, 32.088285039006195, 32.24890815258136, 32.40953126615652, 32.57015437973168, 32.730777493306846, 32.891400606882016, 33.05202372045717, 33.21264683403234, 33.3732699476075, 33.53389306118267])

@njit
def sigmav(te, mQ, nf=3, cg=220.0/27, cf=2.0/9):
   """
   Compute the annihilation cross section for a heavy quark.

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   mQ : float
      Mass of the heavy quark (in GeV)
   nf : int
      Number of light quarks
   cg : float
      Casimir factor for the adjoint representation
   cf : float
      Casimir factor for the fundamental representation

   Returns
   -------
   float
      Annihilation cross section (in GeV^{-2})
   """
   alph = 1/np.interp(np.log10(te), alph0_ergs, alph0_vals)
   return np.pi*alph*alph*(nf*cf+cg)/(16*mQ*mQ)

### Thermal functions ###

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
   Compute the energy of a heavy quark in a plasma

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
   - The energy is the sum of the mass and the (average) thermal energy, making this an approximation
   """
   avg_erg = 3*te
   return np.sqrt(mQ*mQ + avg_erg*avg_erg)


### Evolution of the Universe ###

@njit
def gamma_scaling(te: float) -> float:
   """
   The scaling factor for the entropy density in the Universe.

   Parameters
   ----------
   te : float
      Temperature (in GeV)

   Returns
   -------
   float
      Scaling factor for the entropy density and temperature vs scale factor
   """
   if te > 0:
      lgte = np.log10(te)
      return np.interp(lgte, gdata[:,0], gdata[:,3])
   return 1

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

@njit
def w(te, nQ, mQ):
   """
   Compute the equation of state parameter w of the Universe

   Parameters
   ----------
   te : float
      Temperature (in GeV)
   nQ : float
      Number density of the heavy quark (in GeV^3)
   mQ : float
      Mass of the heavy quark (in GeV)

   Returns
   -------
   float
      Equation of state parameter
   """
   nQeq = n_Q_eq(te, mQ)
   rhoQ = nQ*e_Q(te, mQ)
   h = hubble(te, rhoQ)
   rhoR = rho_SM(te)
   annihilations = sigmav(te, mQ)*(nQ*nQ - nQeq*nQeq)/h
   dhduh = (3*rhoQ + 4*rhoR + annihilations)/(6*pow(M_PLANCK_RED*h,2))
   return 2*dhduh/3 - 1


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
def decayingQs(_: float, y: np.ndarray, mQ: float, qdims: np.ndarray[int], qmult: np.ndarray[int]) -> np.ndarray:
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

   Returns
   -------
   np.ndarray[float]
      Array of the derivatives of the equation values
   """
   if len(y) == 1:
      return np.array([-y[0]/gamma_scaling(y[0])])
   te, nQ = y[0], y[1:]
   nQ_sq = nQ*nQ
   nQeq = qmult*n_Q_eq(te, mQ)
   nQeq_sq = nQeq*nQeq
   eQ = e_Q(te, mQ)
   rhoQ = nQ*eQ
   h = hubble(te, sum(rhoQ))
   s = n_s_SM(te)
   decays = np.array([gammad(mQ, d=d) for d in qdims])*(nQ - nQeq)/h
   annihilations = sigmav(te, mQ)*(nQ_sq - nQeq_sq)/h
   te_eq = [(-te + sum(decays)*eQ/(3*s))/gamma_scaling(te)]
   q_eq = -3*nQ - decays - annihilations
   q_eq = list(q_eq)
   return np.array(te_eq + q_eq)

@njit
def dim_signature(qdims: np.ndarray[int], qmult: np.ndarray[int]) -> tuple[str, np.ndarray[int], np.ndarray[int]]:
   mult_string = ""
   for d0 in range(5,9):
      if d0 in qdims:
         mult_string += "_"+str(qmult[qdims==d0][0])
      else:
         mult_string += "_0"
   return mult_string

def compute_cosmology(qdims: np.ndarray[int], qmult: np.ndarray[int], mQ: float) -> tuple[np.ndarray[float], np.ndarray[float], list[scipy.integrate._ivp.ivp.OdeResult], str]:
   mult_string = dim_signature(qdims, qmult)
   neqs = len(qdims)
   ubreaks, tebreaks, sols = [], [], []
   cont, no_bbn = True, True
   sel0 = neqs*[1]
   te_ini = max(10*mQ, 1e9)
   u1 = 0
   ufin1 = np.log(1e21*(mQ/1e11))
   nQ_ini = n_Q_eq(te_ini, mQ)
   y1 = np.array([te_ini] + [m*nQ_ini for m in qmult])
   while cont:
      sol = solve_ivp(decayingQs, [u1, ufin1], y1, args=(mQ, qdims, qmult,), dense_output=True, events=(crossBBN, dilutedQ,), method='RK45', rtol=1e-7, atol=0)
      try:
         u2 = sol.t_events[0][0]
         # If the BBN event is triggered, we stop the solver
         cont = False
         no_bbn = False
      except IndexError:
         u2 = sol.t[-1]
      try:
         u2 = sol.t_events[1][0]
         # If a dilution event is triggered, we save the result and continue solving the equation
         u1 = u2
         te = sol.y[0][-1]
         tebreaks.append(te)
         nQ = sol.y[1:][:,-1]
         rhoQ = nQ*e_Q(te, mQ)
         rhoSM = rho_SM(te)
         sel0 = [r > 1.01*Q_DILUTION_FAC*rhoSM for r in rhoQ]
         indices = np.where(sel0)
         y1 = np.array([te] + list(nQ[indices]))
         qdims = qdims[indices]
         qmult = qmult[indices]
      except IndexError:
         if no_bbn:
            raise RuntimeError("Stopping solver at u = {:.3f} (T = {:.2e} GeV); this should not happen...".format(sol.t[-1], sol.sol(sol.t[-1])[0]))
      ubreaks.append(u2)
      sols.append(sol)
   return ubreaks, tebreaks, sols, mult_string

def save_cosmology(ubreaks, tebreaks, sols, mQ, output_file, plot=False, verbose=True):
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
      wvals.append(w(te,nQsum,mQ))
      eQ = e_Q(te, mQ)
      lnhvals.append(np.log(hubble(te,nQsum*eQ)))
   res = np.array([uvals, tevals, lnhvals]).T
   bbn_check = 3*wvals[-1]
   # N.B. Do not add a header to the output file; MiMeS cannot handle it
   np.savetxt(output_file, res, fmt="%.9f", delimiter="\t")
   if verbose:
      dimsig = output_file.split("alt_cosmo_")[-1].split("_m")[0]
      print("New model {:s} | afin = {:.2e} | Tfin = {:.1e} GeV | BBN check: {:.3f}".format(dimsig, np.exp(u2), te, bbn_check))
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
   return bbn_check, res

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