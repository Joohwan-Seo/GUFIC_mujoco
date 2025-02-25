import pickle
import matplotlib.pyplot as plt
import tikzplotlib

import os

# Load the data
dt = 0.001

task = "line" # 'regulation', 'line', 'circle'
assert task in ['regulation', 'line', 'circle']

control = 'gufic' #'gufic', 'gic'
assert control in ['gufic', 'gic']

gufic_file = f"data/result_{task}_gufic.pkl" 
gic_file = f"data/result_{task}_gic.pkl"

with open(gufic_file, 'rb') as f:
    data_gufic = pickle.load(f)

with open(gic_file, 'rb') as f:
    data_gic = pickle.load(f)

# print(data_gufic.keys())

Fe_arr_gufic = data_gufic['Fe_arr']
Fd_arr_gufic = data_gufic['Fd_arr']
p_arr_gufic = data_gufic['p_arr']
pd_arr_gufic = data_gufic['pd_arr']
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
Fd_arr_gufic = Fd_arr_gufic[::downsample_factor]

p_arr_gufic = p_arr_gufic[::downsample_factor]
pd_arr_gufic = pd_arr_gufic[::downsample_factor]

tank_force_gufic = tank_force_gufic[::downsample_factor]
tank_impedance_gufic = tank_impedance_gufic[::downsample_factor]

Ff_activation_arr_gufic = Ff_activation_arr_gufic[::downsample_factor]
rho_arr_gufic = rho_arr_gufic[::downsample_factor]

# Do the same thing for gic
Fe_arr_gic = data_gic['Fe_arr']
Fd_arr_gic = data_gic['Fd_arr']

p_arr_gic = data_gic['p_arr']
pd_arr_gic = data_gic['pd_arr']

x_tf_arr_gic = data_gic['x_tf_arr']
x_ti_arr_gic = data_gic['x_ti_arr']

Ff_activation_arr_gic = data_gic['Ff_activation_arr']
rho_arr_gic = data_gic['rho_arr']

tank_force_gic = 0.5 * x_tf_arr_gic**2
tank_impedance_gic = 0.5 * x_ti_arr_gic**2

# Downsample the data_gic by a factor of 20

Fe_arr_gic = Fe_arr_gic[::downsample_factor]
Fd_arr_gic = Fd_arr_gic[::downsample_factor]

p_arr_gic = p_arr_gic[::downsample_factor]
pd_arr_gic = pd_arr_gic[::downsample_factor]

tank_force_gic = tank_force_gic[::downsample_factor]
tank_impedance_gic = tank_impedance_gic[::downsample_factor]

Ff_activation_arr_gic = Ff_activation_arr_gic[::downsample_factor]
rho_arr_gic = rho_arr_gic[::downsample_factor]

t_arr = t_arr[::downsample_factor]



dir = "data"

# plot the force profile 
plt.figure(1)
plt.plot(t_arr,Fe_arr_gufic[:,2])
plt.plot(t_arr,Fe_arr_gic[:,2])
plt.plot(t_arr,-Fd_arr_gufic[:,2])
plt.ylabel('Force z direction')
plt.xlabel('Time (s)')

tikzplotlib.save(f"{dir}/{task}_force_z.tex")

plt.figure(2)
plt.subplot(311)
plt.plot(t_arr,p_arr_gufic[:,0])
plt.plot(t_arr,p_arr_gic[:,0])
plt.plot(t_arr,pd_arr_gufic[:,0])
plt.ylabel('x (m)')
plt.subplot(312)
plt.plot(t_arr,p_arr_gufic[:,1])
plt.plot(t_arr,p_arr_gic[:,1])
plt.plot(t_arr,pd_arr_gufic[:,1])
plt.ylabel('y (m)')
plt.subplot(313)
plt.plot(t_arr,p_arr_gufic[:,2])
plt.plot(t_arr,pd_arr_gic[:,2])
plt.plot(t_arr,pd_arr_gufic[:,2])
plt.ylabel('z (m)')
plt.xlabel('Time (s)')

tikzplotlib.save(f"{dir}/{task}_xyz.tex")

# plot tank values T_f = 0.5 * x_tf^2, T_i = 0.5 * x_ti^2
plt.figure(3)
plt.plot(t_arr,tank_force_gufic)
plt.ylabel('Force Tank Level')
plt.xlabel('Time (s)')

tikzplotlib.save(f"{dir}/{task}_gufic_tank_force.tex")

plt.figure(4)
plt.plot(t_arr,Ff_activation_arr_gufic)
plt.ylabel('Activation of Ff')
plt.xlabel('Time (s)')

plt.figure(5)
plt.plot(t_arr,rho_arr_gufic[:,2])
plt.ylabel('Rho Value z')
plt.xlabel('Time (s)')

tikzplotlib.save(f"{dir}/{task}_gufic_rho_z.tex")

plt.show()