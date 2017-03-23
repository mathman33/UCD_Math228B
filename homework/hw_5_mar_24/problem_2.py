from __future__ import division

import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import scipy.sparse.linalg as LA
import gc as garbage
from tqdm import tqdm
from time import clock,sleep
from operators import make_FV_method
from tabulate import tabulate

def IC(x,choice):
    if choice == 1:
        initial_cond = np.cos(16*np.pi*x)*np.exp(-50*(x-0.5)**2)
    elif choice == 2:
        initial_cond = np.sin(2*np.pi*x)*np.sin(4*np.pi*x)
    elif choice == 3:
        initial_cond = np.piecewise(x, [abs(x-0.5)<0.25, abs(x-0.5)>=0.25], [1, 0])
    return initial_cond

def phi_title(choice):
    if choice == 1:
        return "upwinding"
    elif choice == 2:
        return "Lax-Wendroff"
    elif choice == 3:
        return "Beam-Warming"
    elif choice == 4:
        return "minmod"
    elif choice == 5:
        return "Superbee"
    elif choice == 6:
        return "MC"
    elif choice == 7:
        return "Van Leer"

def IC_title(IC_choice):
    if IC_choice == 1:
        return r"$u_0(x) = \cos(16\pi x)e^{-50(x - 0.5)^2}$    "
    elif IC_choice == 2:
        return r"$u_0(x) = \sin(2\pi x)\sin(4\pi x)$    "
    elif IC_choice == 3:
        return r"$u_0(x) = \mathcal{X}_{(\frac{1}{4},\frac{3}{4})}(x)$    "

def main(IC_choice, phi_choice, power):
    Nt = 10*2**power
    Nx = int(0.9*(Nt-1))
    dt = 1/(Nt-1)
    dx = 1/Nx
    a = 1

    FV_step = make_FV_method(phi_choice,a,dt,dx)

    xaxis = np.linspace(dx/2,1-dx/2,Nx)
    u_initial = IC(xaxis,IC_choice)
    u_0 = u_initial + 0

    for t in xrange((Nt-1)*5):
        u_1 = FV_step(u_0)
        u_0 = u_1 + 0

    abs_diff = abs(u_initial - u_0)
    plt.figure()
    plt.ylim([-1.2,1.2])
    plt.plot(u_initial, "g", lw=5, label=r"$u^0$")
    plt.plot(u_0, "b", lw=2, label=r"$u^{5N}$")
    title = IC_title(IC_choice)
    title += "%s    " % phi_title(phi_choice)
    title += r"$N_t = %d$    " % Nt
    title += r"$N_x = %d$" % Nx
    plt.title(title)
    print title
    plt.legend(loc=0)
    plt.savefig("figures/power_%d/problem_2_%d_%d_a" % (power,IC_choice, phi_choice), dpi=300)
    # plt.show()
    plt.close()
    garbage.collect()

    plt.figure()
    plt.plot(abs_diff, "k", label=r"$\left|u^0 - u^N\right|$")
    title = IC_title(IC_choice)
    title += "%s    " % phi_title(phi_choice)
    title += r"$N_t = %d$    " % Nt
    title += r"$N_x = %d$" % Nx
    plt.title(title)
    print title
    plt.legend(loc=0)
    plt.savefig("figures/power_%d/problem_2_%d_%d_b" % (power,IC_choice, phi_choice), dpi=300)
    # plt.show()
    plt.close()
    garbage.collect()

    # print IC_choice, phi_choice, Nx, dx*norm(abs_diff,1)

if __name__ == "__main__":
    for ic_choice in xrange(1,4):
        # phi_choice: 1 up, 2 LW, 3 BW, 4 minmod, 5 SB, 6 MC, 7 VL
        for phi_choice in xrange(1,8):
            for power in xrange(4,5):
                main(ic_choice,phi_choice,power)

