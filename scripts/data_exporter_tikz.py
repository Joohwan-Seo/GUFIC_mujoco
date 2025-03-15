import pickle
import matplotlib.pyplot as plt
import tikzplotlib # Updated version of mine

import os
import numpy as np

def main(task, save_figure, export_tikz):

    assert task in ['regulation', 'line', 'circle', 'sphere']

    inertia_shaping = False

    gufic_file = f"data/result_{task}_gufic_IS_{inertia_shaping}.pkl" 
    gic_file = f"data/result_{task}_gic_IS_{inertia_shaping}.pkl"

    with open(gufic_file, 'rb') as f:
        data_gufic = pickle.load(f)

    with open(gic_file, 'rb') as f:
        data_gic = pickle.load(f)

    dt = data_gufic['dt']

    Fe_arr_gufic = data_gufic['Fe']
    Fe_raw_arr_gufic = data_gufic['Fe_raw']
    Fd_arr_gufic = data_gufic['Fd']
    p_arr_gufic = data_gufic['p']
    pd_arr_gufic = data_gufic['pd']
    R_arr_gufic = data_gufic['R']
    Rd_arr_gufic = data_gufic['Rd']
    x_tf_arr_gufic = data_gufic['x_tf']
    x_ti_arr_gufic = data_gufic['x_ti']
    rho_arr_gufic = data_gufic['rho']
    Psi_arr_gufic = data_gufic['Psi']

    tank_force_gufic = 0.5 * x_tf_arr_gufic**2
    tank_impedance_gufic= 0.5 * x_ti_arr_gufic**2

    # Do the same thing for gic
    Fe_arr_gic = data_gic['Fe']
    Fe_raw_arr_gic = data_gic['Fe_raw']

    p_arr_gic = data_gic['p']
    pd_arr_gic = data_gic['pd']

    R_arr_gic = data_gic['R']
    Rd_arr_gic = data_gic['Rd']
    Psi_arr_gic = data_gic['Psi']

    # make time array
    N = len(Fe_arr_gufic) # number of time steps
    t_arr = [i*dt for i in range(N)] 

    # Downsample the data_gufic by a factor of 20
    downsample_factor = 20
    Fe_arr_gufic = Fe_arr_gufic[::downsample_factor]
    Fe_raw_arr_gufic = Fe_raw_arr_gufic[::downsample_factor]
    Fd_arr_gufic = Fd_arr_gufic[::downsample_factor]

    p_arr_gufic = p_arr_gufic[::downsample_factor]
    pd_arr_gufic = pd_arr_gufic[::downsample_factor]

    tank_force_gufic = tank_force_gufic[::downsample_factor]
    tank_impedance_gufic = tank_impedance_gufic[::downsample_factor]

    rho_arr_gufic = rho_arr_gufic[::downsample_factor]
    Psi_arr_gufic = Psi_arr_gufic[::downsample_factor]

    # Downsample the data_gic by a factor of 20
    Fe_arr_gic = Fe_arr_gic[::downsample_factor]
    Fe_raw_arr_gic = Fe_raw_arr_gic[::downsample_factor]

    p_arr_gic = p_arr_gic[::downsample_factor]
    pd_arr_gic = pd_arr_gic[::downsample_factor]

    Psi_arr_gic = Psi_arr_gic[::downsample_factor]

    t_arr = t_arr[::downsample_factor]

    # plot the force profile 
    plt.figure(1, figsize = (6,4))
    plt.plot(t_arr,-Fe_arr_gufic[:,2], 'r')
    plt.plot(t_arr,-Fe_arr_gic[:,2], 'b--')
    plt.plot(t_arr,Fd_arr_gufic[:,2], 'k:')
    plt.grid()
    plt.ylabel('$f_z$ direction')
    plt.legend(['GUFIC', 'GIC', 'Desired'])
    plt.xlabel('Time (s)')

    # save figure
    if save_figure:
        plt.savefig(f"data/{task}_force_z.png")

    if export_tikz:
        tikzplotlib.save(f"data/{task}_force_z.tex")

    plt.figure(2, figsize = (6,4.5))
    plt.subplot(311)
    plt.plot(t_arr,p_arr_gufic[:,0], 'r')
    plt.plot(t_arr,p_arr_gic[:,0], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,0], 'k:')
    plt.grid()
    plt.legend(['GUFIC', 'GIC', 'Desired'], loc='upper right', ncols=1 if task == 'circle' else 3)
    plt.ylabel('$x$ (m)')
    plt.subplot(312)
    plt.plot(t_arr,p_arr_gufic[:,1], 'r')
    plt.plot(t_arr,p_arr_gic[:,1], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,1], 'k:')
    plt.grid()
    plt.ylabel('$y$ (m)')
    plt.subplot(313)
    plt.plot(t_arr,p_arr_gufic[:,2], 'r')
    plt.plot(t_arr,p_arr_gic[:,2], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,2], 'k:')
    plt.grid()
    plt.ylabel('$z$ (m)')
    plt.xlabel('Time (s)')

    if save_figure:
        plt.savefig(f"data/{task}_xyz.png")
    if export_tikz:
        tikzplotlib.save(f"data/{task}_xyz.tex")

    plt.figure(3, figsize = (6,6))
    plt.subplot(411)
    plt.plot(t_arr,p_arr_gufic[:,0], 'r')
    plt.plot(t_arr,p_arr_gic[:,0], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,0], 'k:')
    plt.grid()
    plt.legend(['GUFIC', 'GIC', 'Desired'], loc='upper right', ncols=3)
    plt.ylabel('$x$ (m)')
    plt.subplot(412)
    plt.plot(t_arr,p_arr_gufic[:,1], 'r')
    plt.plot(t_arr,p_arr_gic[:,1], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,1], 'k:')
    plt.grid()
    plt.ylabel('$y$ (m)')
    plt.subplot(413)
    plt.plot(t_arr,p_arr_gufic[:,2], 'r')
    plt.plot(t_arr,p_arr_gic[:,2], 'b--')
    plt.plot(t_arr,pd_arr_gufic[:,2], 'k:')
    plt.grid()
    plt.ylabel('$z$ (m)')
    plt.subplot(414)
    plt.plot(t_arr,-Fe_arr_gufic[:,2], 'r')
    plt.plot(t_arr,-Fe_arr_gic[:,2], 'b--')
    plt.plot(t_arr,Fd_arr_gufic[:,2], 'k:')
    plt.grid()
    plt.ylabel('$f_z$ (N)')
    plt.xlabel('Time (s)')

    if save_figure:
        plt.savefig(f"data/{task}_xyz_force.png")
    if export_tikz:
        tikzplotlib.save(f"data/{task}_xyz_force.tex")

    # plot tank values T_f = 0.5 * x_tf^2, T_i = 0.5 * x_ti^2
    plt.figure(4, figsize= (6,4))
    plt.subplot(2,1,1)
    plt.plot(t_arr,tank_force_gufic, 'r')
    plt.grid()
    plt.ylabel('$T_f$')
    plt.subplot(2,1,2)
    plt.plot(t_arr,tank_impedance_gufic, 'r')
    plt.grid()
    plt.ylabel('$T_i$')
    plt.xlabel('Time (s)')

    if save_figure:
        plt.savefig(f"data/{task}_gufic_tank.png")
    if export_tikz:
        tikzplotlib.save(f"data/{task}_gufic_tank.tex")

    plt.figure(5, figsize = (6,4.5))
    plt.plot(t_arr, Psi_arr_gufic, 'r')
    plt.plot(t_arr, Psi_arr_gic, 'b--')
    plt.grid()
    plt.ylabel('$\Psi$')
    plt.legend(['GUFIC', 'GIC'])
    plt.xlabel('Time (s)')

    if save_figure:
        plt.savefig(f"data/{task}_Psi.png")
    if export_tikz:
        tikzplotlib.save(f"data/{task}_Psi.tex")

    # plt.figure(6)
    # plt.plot(t_arr, Psi_so3_gufic, 'r')
    # plt.plot(t_arr, Psi_so3_gic, 'b--')
    # plt.grid()
    # plt.ylabel('$\Psi_{SO(3)}$')
    # plt.legend(['GUFIC', 'GIC'])
    # plt.xlabel('Time (s)')

    plt.show()

if __name__ == "__main__":
    task = "sphere" # 'regulation', 'line', 'circle', 'sphere'
    tasks = ['regulation', 'line', 'circle', 'sphere']

    # tasks = ['sphere']
    for task in tasks:
        main(task = task, save_figure = False, export_tikz = False)