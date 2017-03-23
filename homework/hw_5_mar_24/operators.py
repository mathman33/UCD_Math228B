from __future__ import division

import numpy as np
from numpy import pad
import scipy.sparse as sp
from time import clock

def make_FV_method(phi_choice,a,dt,dx):
    # This function returns a simple iteration technique for
    #   a finite-volume method

    # First, get the flux limiter function based on the user
    #   choice of phi
    phi = get_limiter_function(phi_choice)

    def FV_step(u):
        # Initialize u_next
        u_next = np.zeros(u.shape)

        # Pad the input vector with period data from the other side
        u_padded = pad(u, (2,2), 'wrap')
        # Loop through the points
        for j in xrange(len(u_next)):
            # Get F_{j-1/2} and F_{j+1/2} (numerical flux).
            #   Note that I use j+2 and j+3 because of the padding
            F_j_m_half = get_F_j_m_half(phi,u_padded,j+2,a,dt,dx)
            F_j_p_half = get_F_j_m_half(phi,u_padded,j+3,a,dt,dx)

            # Get u^{n+1} = u^n - (dt/dx)(F_{j+1/2} - F_{j-1/2})
            u_next[j] = u_padded[j+2] - (dt/dx)*(F_j_p_half - F_j_m_half)
        return u_next

    # Return the function
    return FV_step

def get_F_up(u, j, a):
    # The upwinding flux depends on the direction of flow
    if a >= 0:
        return a*u[j-1]
    elif a < 0:
        return a*u[j]

def get_F_j_m_half(phi,u,j,a,dt,dx):
    # This function gets F_{j-1/2} for the given j

    # Get the limited difference delta:
    delta = get_delta(phi,u,j,a)
    # Get the upwinding flux
    F_up = get_F_up(u,j,a)
    # Return the numerical flux
    return F_up + (abs(a)/2)*(1 - abs((a*dt)/(dx)))*delta

def get_delta(phi,u,j,a):
    # This function returns the limited difference delta at point j.
    D = Delta(u,j)
    # If the function is basically flat, |u_j - u_{j-1}|<1e-10, return 0
    if abs(D) < 1e-10:
        return 0
    # Otherwise, get the ratio of jumps across edges, and calculate the
    #   limited flux
    else:
        theta = theta_j_m_half(u, j, a)
        return phi(theta)*D

def Delta(u, j):
    # This is just to match with notation in the homework
    return u[j] - u[j-1]

def theta_j_m_half(u, j, a):
    # This returns the ratio of jumps across edges

    # First get the upwinding direction J_up
    Jup = get_Jup(j,a)
    # Then return the ratio of jumps
    return Delta(u, Jup)/Delta(u, j)

def get_Jup(j,a):
    # Return the appropriate upwinding index for the given flow direction.
    if a >= 0:
        return j-1
    elif a < 0:
        return j+1

def get_limiter_function(phi_choice):
    # Define the flux limiter function based on user choice.
    if phi_choice == 1: # Upwinding
        def phi(theta):
            return 0
    elif phi_choice == 2: # Lax-Wendroff
        def phi(theta):
            return 1
    elif phi_choice == 3: # Beam-Warming
        def phi(theta):
            return theta
    elif phi_choice == 4: # minmod
        def phi(theta):
            return max(0,min(1,theta))
    elif phi_choice == 5: # superbee
        def phi(theta):
            return max(0,min(1,2*theta),min(2,theta))
    elif phi_choice == 6: # MC
        def phi(theta):
            return max(0,min((1+theta)/2,2,2*theta))
    elif phi_choice == 7:# Van Leer
        def phi(theta):
            return (theta + abs(theta))/(1 + abs(theta))
    # return the appropriate function
    return phi

def get_A1_mat(N,K,rho,dx,dt):
    diag = (1 - ((dt**2)*K)/((dx**2)*rho))*np.ones(N)
    diag[0] = 1 - ((dt**2)*K)/(2*(dx**2)*rho)
    diag[-1] = 1 - ((3*(dt**2)*K)/(4*(dx**2)*rho)) \
        - ((dt*np.sqrt(K))/(4*dx*np.sqrt(rho)))
    subdiag = ((dt**2)*K)/(2*(dx**2)*rho)*np.ones(N)
    superdiag = ((dt**2)*K)/(2*(dx**2)*rho)*np.ones(N)
    A1 = np.vstack((superdiag,diag,subdiag))
    A1_mat = sp.dia_matrix((A1,[1,0,-1]),shape=(N,N))
    return A1_mat

def get_A2_mat(N,K,rho,dx,dt):
    diag = (0)*np.ones(N)
    diag[0] = (-dt*K)/(2*dx)
    diag[-1] = (((dt**2)*K*np.sqrt(K))/(4*(dx**2)*np.sqrt(rho))) \
        - ((dt*K)/(4*dx))
    subdiag = (dt*K)/(2*dx)*np.ones(N)
    superdiag = (-dt*K)/(2*dx)*np.ones(N)
    A2 = np.vstack((superdiag,diag,subdiag))
    A2_mat = sp.dia_matrix((A2,[1,0,-1]),shape=(N,N))
    return A2_mat

def get_B1_mat(N,K,rho,dx,dt):
    diag = (0)*np.ones(N)
    diag[0] = (dt)/(2*dx*rho)
    diag[-1] = (((dt**2)*np.sqrt(K))/(4*(dx**2)*rho*np.sqrt(rho))) \
        - ((dt)/(4*dx*rho))
    subdiag = (dt)/(2*dx*rho)*np.ones(N)
    superdiag = (-dt)/(2*dx*rho)*np.ones(N)
    B1 = np.vstack((superdiag,diag,subdiag))
    B1_mat = sp.dia_matrix((B1,[1,0,-1]),shape=(N,N))
    return B1_mat

def get_B2_mat(N,K,rho,dx,dt):
    diag = (1 - ((dt**2)*K)/((dx**2)*rho))*np.ones(N)
    diag[0] = 1 - (3*(dt**2)*K)/(2*(dx**2)*rho)
    diag[-1] = 1 - ((3*(dt**2)*K)/(4*(dx**2)*rho)) \
        - ((dt*np.sqrt(K))/(4*dx*np.sqrt(rho)))
    subdiag = ((dt**2)*K)/(2*(dx**2)*rho)*np.ones(N)
    superdiag = ((dt**2)*K)/(2*(dx**2)*rho)*np.ones(N)
    B2 = np.vstack((superdiag,diag,subdiag))
    B2_mat = sp.dia_matrix((B2,[1,0,-1]),shape=(N,N))
    return B2_mat

def make_LW_method(N,K,rho,dx,dt):
    # Get A_1, A_2, B_1, and B_2
    A1 = get_A1_mat(N,K,rho,dx,dt)
    A2 = get_A2_mat(N,K,rho,dx,dt)
    B1 = get_B1_mat(N,K,rho,dx,dt)
    B2 = get_B2_mat(N,K,rho,dx,dt)

    # Make the Lax-Wendroff recursion step
    def LW_step(p,u):
        p_next = A1.dot(p) + A2.dot(u)
        u_next = B1.dot(p) + B2.dot(u)
        return (p_next,u_next)

    # Return the function
    return LW_step
