from __future__ import division

import numpy as np
import scipy.sparse as sp
from time import clock

def get_upwind_mat(N,nu):
    # Generate the upwinding update matrix of size NxN:
    # 1 - nu on the diagonal, and
    # nu on the sub-diag and upper-right corner (periodicity)
    off_diag = nu*np.ones(N)
    diag = (1-nu)*np.ones(N)
    A = np.vstack((off_diag,diag,off_diag))
    upwind_mat = sp.dia_matrix((A,[N-1,0,-1]),shape=(N,N))
    return upwind_mat

def get_LW_mat(N,nu):
    # Generate the Lax-Wendroff update matrix of size NxN:
    # 1 - nu^2 on the diagonal,
    # (1/2)*(nu + nu^2) on the sub-diag and upper-right corner, and
    # -(1/2)*(nu - nu^2) on the super-diag and bottom-left corner
    diag = (1 - nu**2)*np.ones(N)
    subdiag = (1/2)*(nu + nu**2)*np.ones(N)
    superdiag = -(1/2)*(nu - nu**2)*np.ones(N)
    A = np.vstack((subdiag,superdiag,diag,subdiag,superdiag))
    LW_mat = sp.dia_matrix((A,[N-1,1,0,-1,1-N]),shape=(N,N))
    return LW_mat

def get_CN_mats(N,nu):
    # Generate the "Crank-Nicolson for Advection" LHS and RHS
    #   matrices of size NxN:
    # The LHS matrix has 1 on the diagonal,
    # -nu/4 on the sub-diag and upper-right corner, and
    # nu/4 on the super-diag and bottom-left corner.
    # The RHS matrix is the transpose of the LHS matrix
    diag = np.ones(N)
    pos = (nu/4)*np.ones(N)
    neg = -pos
    A_LHS = np.vstack((neg,pos,diag,neg,pos))
    LHS = sp.dia_matrix((A_LHS,[N-1,1,0,-1,1-N]),shape=(N,N))
    RHS = LHS.transpose()
    return (LHS,RHS)

def make_upwinding_method(N,dx,dt,transport_coef):
    # Construct a function which applies the upwinding method.

    # Get the upwind update matrix
    nu = transport_coef*dt/dx
    A = get_upwind_mat(N,nu)

    def upwind_step(u):
        # Apply the upwind update matrix to the vector u
        return A.dot(u)

    return upwind_step

def make_LW_method(N,dx,dt,transport_coef):
    # Construct a function which applies the Lax-Wendroff method.

    # Get the Lax-Wendroff update matrix
    nu = transport_coef*dt/dx
    A = get_LW_mat(N,nu)

    def LW_step(u):
        # Apply the Lax-Wendroff update matrix to the vector u
        return A.dot(u)

    return LW_step

def make_CN_method(N,dx,dt,transport_coef):
    # Construct a function which applies the "Crank-Nicolson
    #   for Advection" method.

    # Get the "Crank Nicolson for Advection" LHS and RHS matrices
    nu = transport_coef*dt/dx
    (LHS, RHS) = get_CN_mats(N,nu)

    def CN_step(u):
        # Apply the RHS CN matrix to the vector u
        right_side = RHS.dot(u)
        # Solve LHS u^(n+1) = RHS u^n for u^(n+1)
        return sp.linalg.spsolve(LHS,right_side)

    return CN_step
