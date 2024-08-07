# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:46:09 2021

@author: Vaisakh, Sebastian
"""
import os
import sys
import contextlib

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.integrate import odeint
import collections
from fractions import Fraction
#from scipy import stats
# import seaborn as sns

# Turn off warnings from odeint
chatter = 0

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


repdict = dict({ 1: [Fraction(3),Fraction(1),Fraction(-1,3)], 2: [Fraction(3),Fraction(1),Fraction(2,3)], 3: [Fraction(3),Fraction(2),Fraction(1,6)], 4: [Fraction(3), Fraction(2), Fraction(-5,6)], 5: [Fraction(3), Fraction(2), Fraction(7,6)],
                 6: [Fraction(3), Fraction(3), Fraction(-1,3)], 7: [Fraction(3), Fraction(3), Fraction(2,3)], 8: [Fraction(3), Fraction(3), Fraction(-4,3)], 9:[Fraction(6), Fraction(1), Fraction(-1,3)], 10: [Fraction(6), Fraction(1), Fraction(2,3)],
                11: [Fraction(6), Fraction(2), Fraction(1,6)], 12: [Fraction(8), Fraction(1), Fraction(-1)], 13: [Fraction(8), Fraction(2), Fraction(-1,2)], 14: [Fraction(15), Fraction(1), Fraction(-1,3)], 15: [Fraction(15), Fraction(1), Fraction(2,3)],
                16: [Fraction(3), Fraction(3), Fraction(5,3)], 17: [Fraction(3), Fraction(4), Fraction(1,6)], 18: [Fraction(3), Fraction(4), Fraction(-5,6)], 19: [Fraction(3), Fraction(4), Fraction(7,6)], 20: [Fraction(15), Fraction(2), Fraction(1,6)]
})


def ENcalc(summed, subbed=[]):
    E = 0
    N = 0
    for rep in summed:
        E = E + rep[0]*rep[1]*((rep[1]**2 - 1)/12 + rep[2]**2)
        N = N + rep[1]*dynkind(3,rep[0])
    if len(subbed)>0:
        for rep in subbed:
            E = E - rep[0]*rep[1]*((rep[1]**2 - 1)/12 + rep[2]**2)
            N = N - rep[1]*dynkind(3,rep[0])
    return E, N

def dynkind(N, r):
    dyn=[collections.defaultdict(lambda : r**2),{1:0,2:1/2,3:2,4:5},{1:0,3:1/2,6:5/2,8:3,10:15/2,15:10}]
    return Fraction(dyn[N-1][r])

def casimir(N, r):
    cas=[collections.defaultdict(lambda : r**2),{1:0,2:3/4,3:2,4:15/4},{1:0,3:4/3,6:10/3,8:3,10:6,15:16/3}]
    return cas[N-1][r]

def casadj(i):
    return 0 if i==1 else i

def do_it(f, m_Q=5e11):
    n_g = 3
    a_SM = [4/3*n_g+1/10,-(22/3-4/3*n_g-1/6),-(11-4/3*n_g)]
    b1 = np.array([[0,0,0],[0,136/3,0],[0,0,102]])
    b2 = n_g*np.array([[19/15,1/5,11/30],[3/5,49/3,3/2],[44/15,4,76/3]])
    b3 = np.array([[9/50,3/10,0],[9/10,13/6,0],[0,0,0]])
    b_SM = -(b1-b2-b3)
    b_SM = b_SM.transpose()
    a, b = extend(f)
    al1, al2, al3, t = solve_n_plot(a,b, a_SM, b_SM, m_Q)
    LP1, LP2, LP3 = results(al1, al2, al3, t)
    return LP1, LP2, LP3

def extend(f_bSM):
    a_bSM=[0,0,0]
    b_bSM=np.zeros((3,3))
    f_bSM = np.array(f_bSM).astype(float)
    f_bSM[0] = np.sqrt(3/5)*f_bSM[0]
    kappa = 1
    for i in range(len(f_bSM[0])):
        a_bSM[0] = a_bSM[0] + 4/3*kappa*dynkind(1,f_bSM[0][i])*f_bSM[1][i]*f_bSM[2][i]
        a_bSM[1] = a_bSM[1] + 4/3*kappa*dynkind(2,f_bSM[1][i])*f_bSM[2][i]
        a_bSM[2] = a_bSM[2] + 4/3*kappa*dynkind(3,f_bSM[2][i])*f_bSM[1][i]
        b_bSM[0][0] = b_bSM[0][0] + kappa*dynkind(1,f_bSM[0][i])*(4*casimir(1,f_bSM[0][i]) + 20/3*casadj(1))*f_bSM[1][i]*f_bSM[2][i]
        b_bSM[1][1] = b_bSM[1][1] + kappa*dynkind(2,f_bSM[1][i])*(4*casimir(2,f_bSM[1][i]) + 20/3*casadj(2))*f_bSM[2][i] #
        b_bSM[2][2] = b_bSM[2][2] + kappa*dynkind(3,f_bSM[2][i])*(4*casimir(3,f_bSM[2][i]) + 20/3*casadj(3))*f_bSM[1][i] #
        b_bSM[0][1] = b_bSM[0][1] + 4*kappa*casimir(2,f_bSM[1][i])*dynkind(1,f_bSM[0][i])*f_bSM[1][i]*f_bSM[2][i]
        b_bSM[1][0] = b_bSM[1][0] + 4*kappa*casimir(1,f_bSM[0][i])*dynkind(2,f_bSM[1][i])*f_bSM[2][i] #
        b_bSM[0][2] = b_bSM[0][2] + 4*kappa*casimir(3,f_bSM[2][i])*dynkind(1,f_bSM[0][i])*f_bSM[1][i]*f_bSM[2][i]
        b_bSM[2][0] = b_bSM[2][0] + 4*kappa*casimir(1,f_bSM[0][i])*dynkind(3,f_bSM[2][i])*f_bSM[1][i] #
        b_bSM[1][2] = b_bSM[1][2] + 4*kappa*casimir(3,f_bSM[2][i])*dynkind(2,f_bSM[1][i])*f_bSM[2][i] #
        b_bSM[2][1] = b_bSM[2][1] + 4*kappa*casimir(2,f_bSM[1][i])*dynkind(3,f_bSM[2][i])*f_bSM[1][i] #
    return a_bSM, b_bSM

def solve_n_plot(a_bSM,b_bSM, a_SM, b_SM, m_Q = 5e11):
    def solver(y, t, a_SM, b_SM, a_bSM, b_bSM):
#        m_Q = 5*10**11 #heavy fermion mass
        if t < np.log(m_Q/91.188)/(2*np.pi):
            a = a_SM
            b = b_SM
        else:
            a = a_SM + a_bSM
            b = b_SM + b_bSM
        dydt = -a-np.matmul(b,1/y)/(4*np.pi)
        return dydt
    t = np.linspace(0,20,100000)
    y0=np.array([1/0.016923, 1/0.03374, 1/0.1173]) #\alpha^{-1} ar m_Z = 91.188 GeV
    if chatter==0:
        with stdout_redirected():
            test = odeint(solver, y0, t, args=(np.array(a_SM),np.array(b_SM),np.array(a_bSM),np.array(b_bSM)))
    else:
        test = odeint(solver, y0, t, args=(np.array(a_SM),np.array(b_SM),np.array(a_bSM),np.array(b_bSM)))
    alpha1=1/test[:, 0]
    alpha2=1/test[:, 1]
    alpha3=1/test[:, 2]
    return alpha1, alpha2, alpha3, t

def results(alpha1, alpha2, alpha3, t):
    mZ = 91.188 #GeV
    mu=mZ*np.exp(2*np.pi*t)
    g1=np.sqrt(4*np.pi*alpha1)
    g2=np.sqrt(4*np.pi*alpha2)
    g3=np.sqrt(4*np.pi*alpha3)
    # threshold=1000                       #arbitrary high threshold for coupling above which LP
    # return np.floor(np.log10(mu[np.where(1/g1==0)[0][0]])).astype(int), np.floor(np.log10(mu[np.where(1/g2==0)[0][0]])).astype(int), np.floor(np.log10(mu[np.where(1/g3==0)[0][0]])).astype(int)
    return np.array([mu[np.where(1/g==0)[0][0]] for g in [g1,g2,g3]])
