from __future__ import division

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import scipy.sparse.linalg as LA
import gc as garbage
from tqdm import tqdm
from time import clock,sleep
from operators import make_upwinding_method, make_LW_method
from tabulate import tabulate

def refinement_study(Nxs,dxs,initials,finals,times):
    ones = []
    twos = []
    maxs = []

    for i in xrange(len(initials)):
        ones.append(dxs[i]*norm(initials[i]-finals[i],1))
        twos.append(dxs[i]*norm(initials[i]-finals[i],2))
        maxs.append(norm(initials[i]-finals[i],np.inf))

    norms = [ones,twos,maxs]
    for N in norms:
        ratios = [0] + [N[i]/N[i-1] for i in xrange(1,len(N))]
        tabulate_format = [[Nxs[i],N[i],ratios[i]] for i in xrange(len(N))]
        print tabulate(tabulate_format,headers=["N_x","norm","ratio"],tablefmt="latex")

    time_ratios = [0] + [times[i]/times[i-1] for i in xrange(1,len(N))]
    tabulate_format = [[Nxs[i],times[i],time_ratios[i]] for i in xrange(len(times))]
    print tabulate(tabulate_format,headers=["N_x", "time (in seconds)", "ratio of times"], tablefmt="latex")

def smooth_initial_condition(x):
    return (1/2)*np.sin(2*np.pi*x)+(1/2)

def discon_initial_condition(x):
    return np.piecewise(x, [abs(x-0.5)<0.25, abs(x-0.5)>=0.25], [1, 0])

def main():
    max_power = 5
    Nts = [10*2**(i)+1 for i in xrange(4,max_power)]
    Nxs = [int(0.9*(Nt-1)) for Nt in Nts]
    dts = [1/(Nt-1) for Nt in Nts]
    dxs = [1/Nx for Nx in Nxs]
    transport_coef = 1

    initials = []
    finals =[]

    solutions = []
    times = []

    for Nt in tqdm(Nts):
        tic = clock()
        Nx = int(0.9*Nt)
        dx = 1/Nx
        dt = 1/Nt

        # step = make_upwinding_method(Nx,dx,dt,transport_coef)
        step = make_LW_method(Nx,dx,dt,transport_coef)

        xaxis = np.linspace(0,1-dx,Nx)
        # u_0 = smooth_initial_condition(xaxis)
        u_0 = discon_initial_condition(xaxis)

        solution_mat = np.zeros((Nx,Nt+1))
        solution_mat[:,0] = u_0

        initials.append(u_0)

        for t in tqdm(xrange(1,Nt+1)):
            u_1 = step(u_0)
            solution_mat[:,t] = u_1
            u_0 = u_1 + 0

        toc = clock()

        # color_choice = cm.terrain
        # colorbar_min = -0.4
        # colorbar_max = 1.4
        # axes = plt.figure().add_subplot(111)
        # plt.imshow(solution_mat,interpolation="nearest", aspect="auto",vmin=colorbar_min,vmax=colorbar_max,cmap=color_choice)
        # plt.colorbar()
        # plt.xlabel("Timesteps (Final Time = 1)")
        # plt.ylabel(r"$j$ in $x_j$")
        # # plt.show()
        # # plt.title(r"Upwinding    $N_x = %d$    $u_0(x) = (1/2)\sin(2\pi x) + (1/2)$" % (Nx))
        # # plt.savefig("figures/upwind_smooth.png", type="png", dpi=300)
        # plt.title(r"Lax-Wendroff    $N_x = %d$    $u_0(x) = (1/2)\sin(2\pi x) + (1/2)$" % (Nx))
        # plt.savefig("figures/LW_smooth.png", type="png", dpi=300)
        # # plt.title(r"Upwinding    $N_x = %d$    $u_0(x) = \mathcal{X}_{(1/4,3/4)}$" % (Nx))
        # # plt.savefig("figures/upwind_discon.png", type="png", dpi=300)
        # # plt.title(r"Lax-Wendroff    $N_x = %d$    $u_0(x) = \mathcal{X}_{(1/4,3/4)}$" % (Nx))
        # # plt.savefig("figures/LW_discon.png", type="png", dpi=300)
        # plt.close()
        # garbage.collect()

        plt.figure()
        plt.plot(solution_mat[:,0], "k", label=r"$t = 0$")
        plt.plot(solution_mat[:,round((1/3)*Nt)], "g", label=r"$t = \frac{1}{3}$")
        plt.plot(solution_mat[:,round((2/3)*Nt)], "b", label=r"$t = \frac{2}{3}$")
        plt.plot(solution_mat[:,Nt], "r", label=r"$t = 1$")
        plt.xlim(0,144)
        plt.legend(loc=0)
        plt.xlabel(r"$x_j$")
        plt.ylabel(r"$u(x_j)$",rotation=0)
        # plt.show()
        # plt.title(r"Upwinding    $N_x = %d$    $u_0(x) = (1/2)\sin(2\pi x) + (1/2)$" % (Nx))
        # plt.savefig("figures/upwind_smooth_snapshots.png", type="png", dpi=300)
        # plt.title(r"Lax-Wendroff    $N_x = %d$    $u_0(x) = (1/2)\sin(2\pi x) + (1/2)$" % (Nx))
        # plt.savefig("figures/LW_smooth_snapshots.png", type="png", dpi=300)
        # plt.title(r"Upwinding    $N_x = %d$    $u_0(x) = \mathcal{X}_{(1/4,3/4)}$" % (Nx))
        # plt.savefig("figures/upwind_discon_snapshots.png", type="png", dpi=300)
        plt.title(r"Lax-Wendroff    $N_x = %d$    $u_0(x) = \mathcal{X}_{(1/4,3/4)}$" % (Nx))
        plt.savefig("figures/LW_discon_snapshots.png", type="png", dpi=300)
        plt.close()
        garbage.collect()

        finals.append(u_1)

        # solutions.append(solution_mat)
        times.append(toc - tic)

    # refinement_study(Nxs,dxs,initials,finals,times)

if __name__ == "__main__":
    main()