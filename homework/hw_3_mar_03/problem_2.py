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
from operators import make_PR_step_method, make_RK2_step_method

def initial_condition_b(x,y):
    return (np.exp(-100*(x**2 + y**2)), 0*x)

def initial_condition_c(x,y):
    return (1 - 2*x, 0.05*y)

def make_RHS_ODEs(a,I,gamma,epsilon):
    def f_v(v,w):
        return (a - v)*(v - 1)*v - w + I

    def f_w(v,w):
        return epsilon*(v - gamma*w)

    return (f_v,f_w)

def main():
    dx = 2**(-7)
    final_time = 600
    dt = 1
    Nx = int(round(1/dx))
    Nt = int(round(final_time/dt))
    D = 5*(10**(-5))
    a = 0.1
    gamma = 2
    epsilon = 0.005
    I = 0

    (f_v,f_w) = make_RHS_ODEs(a,I,gamma,epsilon)

    PR_step = make_PR_step_method(Nx,dx,dt/2,D)
    RK2_step = make_RK2_step_method(Nx,dx,dt,f_v,f_w)

    xaxis = np.linspace(0,1-dx,Nx) + dx/2
    yaxis = np.linspace(0,1-dx,Nx) + dx/2
    x,y = np.meshgrid(xaxis,yaxis)
    # (v_0, w_0) = initial_condition_b(x,y)
    (v_0, w_0) = initial_condition_c(x,y)

    color_choice = cm.Blues
    colorbar_min = -0.25
    colorbar_max = 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0,1)
    plt.ion()
    frame = ax.plot_surface(x,y,v_0,cmap=color_choice,vmin=colorbar_min,vmax=colorbar_max,cstride=5,rstride=5)
    fig.colorbar(frame)
    # ax.view_init(30,0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$v$")
    text = ax.text2D(0.05,0.95,r"$t = 0$",transform=ax.transAxes)
    # plt.savefig("figures/problem_2c_0.png", type="png", dpi=300)
    plt.pause(0.01)

    for index, t in tqdm(enumerate(xrange(1,Nt+1))):
        # Half step PR
        v_star = PR_step(v_0)
        w_star = w_0 + 0

        # Full step RK2
        (v_double_star, w_double_star) = RK2_step(v_star, w_star)

        # Half step PR
        v_1 = PR_step(v_double_star)
        w_1 = w_double_star + 0

        # Redefine v_0 and w_0
        v_0 = v_1 + 0
        w_0 = w_1 + 0
        if t in [10,25,50,100,150,200,250,300,350,400,450,500,550,600]:
            ax.collections.remove(frame)
            for txt in ax.texts:
                txt.set_visible(False)
            frame = ax.plot_surface(x,y,v_1,cmap=color_choice,vmin=colorbar_min,vmax=colorbar_max,cstride=5,rstride=5)
            # ax.view_init(30,-t)
            text = ax.text2D(0.05,0.95,r"$t = %.3f$" % (t*dt),transform=ax.transAxes)
            # plt.savefig("figures/problem_2c_%d.png" % (index+1), type="png", dpi=300)
            plt.pause(0.01)

if __name__ == "__main__":
    main()

