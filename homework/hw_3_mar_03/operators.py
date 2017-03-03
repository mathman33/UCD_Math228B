from __future__ import division

import numpy as np
import scipy.sparse as sp
from time import clock

def get_Lap_1D(N,dx):
    off_diag = np.ones(N)
    diag = (-2)*np.ones(N)
    diag[0] = -1
    diag[-1] = -1
    A = np.vstack((off_diag,diag,off_diag))/(dx**2)
    L = sp.dia_matrix((A,[-1,0,1]),shape=(N,N))
    return L

def make_PR_step_method(N,dx,dt,transport_coef):
    
    L = get_Lap_1D(N,dx)
    I = sp.identity(N)
    right_mat = I + (transport_coef*dt/2)*L
    left_mat = sp.csc_matrix(I - (transport_coef*dt/2)*L)

    def PR_step(u):
        RHS_half = right_mat.dot(u)
        u_half = sp.linalg.spsolve(left_mat, RHS_half)

        RHS = right_mat.dot(np.transpose(u_half))
        u_np1 = np.transpose(sp.linalg.spsolve(left_mat, RHS))

        return u_np1
    return PR_step

def make_RK2_step_method(N,dx,dt,f_v,f_w):

    def RK2_step(v,w):
        k1 = f_v(v,w)
        l1 = f_w(v,w) # left-hand slopes
        
        k2 = f_v(v+dt*k1,w+dt*l1)
        l2 = f_w(v+dt*k1,w+dt*l1) # right-hand slopes
        
        k = (k1+k2)/2 # average v-slope
        l = (l1+l2)/2 # average w-slope 
        v_new = v + dt*k
        w_new = w + dt*l
        return (v_new, w_new)

    return RK2_step

