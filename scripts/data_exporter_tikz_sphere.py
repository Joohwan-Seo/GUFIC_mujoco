import pickle
import matplotlib.pyplot as plt
import tikzplotlib # Updated version of mine

import numpy as np

import os

# Load the data
dt = 0.001

task = "sphere" # 'regulation', 'line', 'circle'

control = 'gufic' #'gufic', 'gic'
assert control in ['gufic', 'gic']

inertia_shaping = False

export_tikz = True

save_figure = False

gufic_file = f"data/result_{task}_gufic_IS_{inertia_shaping}.pkl" 
# gic_file = f"data/result_{task}_gic_IS_{inertia_shaping}.pkl"

with open(gufic_file, 'rb') as f:
    data_gufic = pickle.load(f)

# print(data_gufic.keys())

Fe_arr_gufic = data_gufic['Fe_arr']
Fe_raw_arr_gufic = data_gufic['Fe_raw_arr']
Fd_arr_gufic = data_gufic['Fd_arr']
p_arr_gufic = data_gufic['p_arr']
pd_arr_gufic = data_gufic['pd_arr']
R_arr_gufic = data_gufic['R_arr']
Rd_arr_gufic = data_gufic['Rd_arr']
x_tf_arr_gufic = data_gufic['x_tf_arr']
x_ti_arr_gufic = data_gufic['x_ti_arr']
Ff_activation_arr_gufic = data_gufic['Ff_activation_arr']
rho_arr_gufic = data_gufic['rho_arr']

tank_force_gufic = 0.5 * x_tf_arr_gufic**2
tank_impedance_gufic= 0.5 * x_ti_arr_gufic**2

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

Ff_activation_arr_gufic = Ff_activation_arr_gufic[::downsample_factor]
rho_arr_gufic = rho_arr_gufic[::downsample_factor]

t_arr = t_arr[::downsample_factor]

# Calculate the error function
Psi_arr = np.zeros((p_arr_gufic.shape[0],))
rot_psi_arr = np.zeros((p_arr_gufic.shape[0],))
euc_err_arr = np.zeros((p_arr_gufic.shape[0],))
for i in range(p_arr_gufic.shape[0]):
    euc_err_arr[i] = 0.5 * np.linalg.norm(p_arr_gufic[i] - pd_arr_gufic[i])**2
    rot_psi_arr[i] = np.trace(np.eye(3) - Rd_arr_gufic[i].T @ R_arr_gufic[i])
    Psi_arr[i] = euc_err_arr[i] + rot_psi_arr[i]


dir = "data"

# plot the force profile 
plt.figure(1, figsize = (6,4))
plt.plot(t_arr,-Fe_arr_gufic[:,2], 'r')
plt.plot(t_arr,Fd_arr_gufic[:,2], 'k:')
plt.grid()
plt.ylabel('$f_z$ direction')
plt.legend(['GUFIC','Desired'])
plt.xlabel('Time (s)')

# save figure
if save_figure:
    plt.savefig(f"{dir}/{task}_force_z.png")

if export_tikz:
    tikzplotlib.save(f"{dir}/{task}_force_z.tex")

plt.figure(2, figsize = (6,4.5))
plt.subplot(311)
plt.plot(t_arr,p_arr_gufic[:,0], 'r')
plt.plot(t_arr,pd_arr_gufic[:,0], 'k:')
plt.grid()
plt.legend(['GUFIC', 'Desired'], loc='upper right', ncols=1 if task == 'circle' else 3)
plt.ylabel('$x$ (m)')
plt.subplot(312)
plt.plot(t_arr,p_arr_gufic[:,1], 'r')
plt.plot(t_arr,pd_arr_gufic[:,1], 'k:')
plt.grid()
plt.ylabel('$y$ (m)')
plt.subplot(313)
plt.plot(t_arr,p_arr_gufic[:,2], 'r')
plt.plot(t_arr,pd_arr_gufic[:,2], 'k:')
plt.grid()
plt.ylabel('$z$ (m)')
plt.xlabel('Time (s)')

if save_figure:
    plt.savefig(f"{dir}/{task}_xyz.png")
if export_tikz:
    tikzplotlib.save(f"{dir}/{task}_xyz.tex")

plt.figure(3, figsize = (6,6))
plt.subplot(411)
plt.plot(t_arr,p_arr_gufic[:,0], 'r')
plt.plot(t_arr,pd_arr_gufic[:,0], 'k:')
plt.grid()
plt.legend(['GUFIC', 'Desired'], loc='upper right', ncols=1 if task == 'circle' else 3)
plt.ylabel('$x$ (m)')
plt.subplot(412)
plt.plot(t_arr,p_arr_gufic[:,1], 'r')
plt.plot(t_arr,pd_arr_gufic[:,1], 'k:')
plt.grid()
plt.ylabel('$y$ (m)')
plt.subplot(413)
plt.plot(t_arr,p_arr_gufic[:,2], 'r')
plt.plot(t_arr,pd_arr_gufic[:,2], 'k:')
plt.grid()
plt.ylabel('$z$ (m)')
plt.subplot(414)
plt.plot(t_arr,-Fe_arr_gufic[:,2], 'r')
plt.plot(t_arr,Fd_arr_gufic[:,2], 'k:')
plt.grid()
plt.ylabel('$f_z$ (N)')
plt.xlabel('Time (s)')

if save_figure:
    plt.savefig(f"{dir}/{task}_xyz_force.png")
if export_tikz:
    tikzplotlib.save(f"{dir}/{task}_xyz_force.tex")

# plot tank values T_f = 0.5 * x_tf^2, T_i = 0.5 * x_ti^2
plt.figure(4, figsize= (6,4))
plt.subplot(2,1,1)
plt.plot(t_arr,tank_force_gufic, 'r')
plt.grid()
plt.ylabel('Force Tank Level')
plt.subplot(2,1,2)
plt.plot(t_arr,tank_impedance_gufic, 'r')
plt.grid()
plt.ylabel('Impedance Tank Level')
plt.xlabel('Time (s)')

if save_figure:
    plt.savefig(f"{dir}/{task}_gufic_tank.png")
if export_tikz:
    tikzplotlib.save(f"{dir}/{task}_gufic_tank.tex")

plt.figure(5, figsize = (6,4))
plt.plot(t_arr, Psi_arr, 'r')
plt.grid()
plt.ylabel('Error Function')
plt.xlabel('Time (s)')

if save_figure:
    plt.savefig(f"{dir}/{task}_gufic_error_func.png")
if export_tikz:
    tikzplotlib.save(f"{dir}/{task}_gufic_error_func.tex")

plt.show()