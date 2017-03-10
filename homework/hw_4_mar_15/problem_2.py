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
from time import clock
from operators import make_CN_method

def discon_initial_condition(x):
    return np.piecewise(x, [abs(x-0.5)<0.25, abs(x-0.5)>=0.25], [1, 0])

def smooth_initial_condition(x):
    return (1/2)*np.sin(10*np.pi*x)+(1/2)

def main():
    max_power = 5
    Nts = [10*2**(i) for i in xrange(4,max_power)]
    Nxs = [int(0.9*Nt) for Nt in Nts]
    dts = [1/Nt for Nt in Nts]
    dxs = [1/Nx for Nx in Nxs]
    transport_coef = 1

    initial_sums = []
    final_sums = []
    times = []
    finals = []
    solutions =[]

    for Nt in tqdm(Nts):
        tic = clock()
        Nx = int(0.9*Nt)
        dx = 1/Nx
        dt = 1/Nt

        step = make_CN_method(Nx,dx,dt,transport_coef)

        xaxis = np.linspace(0,1-dx,Nx)
        u_0 = discon_initial_condition(xaxis)
        # u_0 = smooth_initial_condition(xaxis)

        solution_mat = np.zeros((Nx,Nt+1))
        solution_mat[:,0] = u_0

        initial_sums.append(np.sum(u_0))

        for t in tqdm(xrange(1,Nt+1)):
            u_1 = step(u_0)
            solution_mat[:,t] = u_1
            u_0 = u_1 + 0

        toc = clock()

        color_choice = cm.terrain
        colorbar_min = -0.4
        colorbar_max = 1.4
        axes = plt.figure().add_subplot(111)
        plt.imshow(solution_mat,interpolation="nearest", aspect="auto",vmin=colorbar_min,vmax=colorbar_max,cmap=color_choice)
        plt.colorbar()
        plt.xlabel("Timesteps (Final Time = 1)")
        plt.ylabel(r"$j$ in $x_j$")
        plt.title(r"Crank-Nicolson for Advection    $N_x = %d$    $u_0(x) = \mathcal{X}_{(1/4,3/4)}$" % (Nx))
        # plt.show()
        # plt.savefig("figures/CN_smooth.png", type="png", dpi=300)
        # plt.savefig("figures/CN_discon.png", type="png", dpi=300)
        plt.close()
        garbage.collect()

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
        # plt.savefig("figures/CN_smooth_snapshots.png", type="png", dpi=300)
        plt.savefig("figures/CN_snapshots.png", type="png", dpi=300)
        plt.close()
        garbage.collect()

        finals.append(u_1)
        solutions.append(solution_mat)
        final_sums.append(np.sum(u_1))
        times.append(toc - tic)


if __name__ == "__main__":
    main()