from __future__ import division

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import scipy.sparse.linalg as LA
import gc as garbage
from tqdm import tqdm
from time import clock
from operators import make_PR_step_method

def initial_condition(x,y):
    return np.exp(-10*((x - 0.3)**2 + (y - 0.4)**2))

def refinement_study(solutions):
    restrictions = []
    for i, sol in enumerate(solutions[1:]):
        print sol.shape
        next
        coarse_sol = solutions[i]
        restriction = np.zeros(coarse_sol.shape)
        for I_ind, I in enumerate(xrange(1, sol.shape[0], 3)):
            for J_ind, J in enumerate(xrange(1, sol.shape[0], 3)):
                restriction[I_ind][J_ind] = sol[I][J]
        DIFF = restriction - coarse_sol
        print np.max(DIFF)

def main():
    max_power = 7
    dxs = [3**(-i) for i in xrange(1,max_power)]
    transport_coef = 0.1

    initials = []
    finals = []
    times = []
    solutions =[]

    for dx in tqdm(dxs):
        tic = clock()
        dt = dx
        Nx = int(round(1/dx))
        Nt = int(round(1/dt))

        PR_step = make_PR_step_method(Nx,dx,dt,transport_coef)

        xaxis = np.linspace(0,1-dx,Nx) + dx/2
        yaxis = np.linspace(0,1-dx,Nx) + dx/2
        x,y = np.meshgrid(xaxis,yaxis)
        u_0 = initial_condition(x,y)

        initials.append(np.sum(u_0))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_zlim(0,1)
        # plt.ion()
        # frame = ax.plot_surface(x,y,u_0,cstride=4,rstride=4)
        # ax.view_init(30,0)
        # ax.set_xlabel(r"$x$")
        # ax.set_ylabel(r"$y$")
        # ax.set_zlabel(r"$u$")
        # text = ax.text2D(0.05,0.95,r"$t = 0$",transform=ax.transAxes)
        # plt.pause(0.05)

        for t in tqdm(xrange(1,Nt+1)):
            u_1 = PR_step(u_0)
            u_0 = u_1 + 0
            # ax.collections.remove(frame)
            # for txt in ax.texts:
            #     txt.set_visible(False)
            # ax.view_init(30,t*5)
            # frame = ax.plot_surface(x,y,u_1.todense(),cstride=4,rstride=4)
            # text = ax.text2D(0.05,0.95,r"$t = %.3f$" % (t*dt),transform=ax.transAxes)
            # plt.pause(0.05)
        # plt.close()
        # garbage.collect()
        solutions.append(u_1)
        finals.append(np.sum(u_1))
        toc = clock()
        times.append(toc - tic)

    differences = [abs(initials[i] - finals[i]) for i in xrange(len(initials))]
    print "initials\n",initials
    print "finals\n",finals
    print "differences\n",differences
    print "times\n",times

    refinement_study(solutions)

if __name__ == "__main__":
    main()