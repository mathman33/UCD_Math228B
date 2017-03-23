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
from operators import make_LW_method
from tabulate import tabulate

def IC(x,IC_choices):
    initial_cond = []
    for choice in IC_choices:
        if choice == 1:
            initial_cond.append(np.cos(32*np.pi*x)*np.exp(-150*(x-0.5)**2))
        elif choice == 2:
            initial_cond.append(np.sin(2*np.pi*x)*np.sin(4*np.pi*x))
        elif choice == 3:
            initial_cond.append(np.piecewise(x, [abs(x-0.25)<0.05, abs(x-0.75)<0.05], [1, 1]))
    return initial_cond

def main():
    IC_choices = [1,1]
    power = 5

    Nt = 10*2**power
    Nx = int(0.9*(Nt-1))
    dt = 1/(Nt-1)
    dx = 1/Nx
    K = 0.3
    rho = 1

    LW_step = make_LW_method(Nx,K,rho,dx,dt)

    xaxis = np.linspace(dx/2,1-dx/2,Nx)
    [p_0,u_0] = IC(xaxis,IC_choices)

    plt.figure()
    plt.title(r"Timestep $n = 0$, Time $t = 0$")
    plt.plot(p_0,label=r"$p_j^0$")
    plt.plot(u_0,label=r"$u_j^0$")
    plt.ylim([-1.5,1.5])
    plt.legend(loc=2)
    plt.pause(0.0001)
    # plt.savefig("figures/problem_1_c_000.png", dpi=300)
    plt.close()

    for t in xrange(int(5*Nt)):
        (p_1,u_1) = LW_step(p_0,u_0)
        p_0 = p_1 + 0
        u_0 = u_1 + 0
        if t % 10 == 0:
            plt.figure()
            plt.title(r"Timestep $n = %d$ ,Time $t = %.3f$" % (t,t*dt))
            plt.plot(p_0,label=r"$p_j^{%d}$" % t)
            plt.plot(u_0,label=r"$u_j^{%d}$" % t)
            plt.ylim([-1.5,1.5])
            plt.legend(loc=2)
            plt.pause(0.0001)
            # plt.savefig("figures/problem_1_c_%.3d" % int(t/10), dpi=300)
            plt.close()

    plt.figure()
    plt.plot(p_0)
    plt.plot(u_0)
    # plt.ylim([-1.5,1.5])
    plt.pause(0.0001)
    plt.show()
    plt.close()
    garbage.collect()

if __name__ == "__main__":
    main()
