import pickle
import matplotlib.pyplot as plt

# Load the data
dt = 0.001

task = "line" # 'regulation', 'line', 'circle'
assert task in ['regulation', 'line', 'circle']

control = 'gufic' #'gufic', 'gic'
assert control in ['gufic', 'gic']

dir_file = f"data/result_{task}_{control}.pkl" 

with open(dir_file, 'rb') as f:
    data = pickle.load(f)

# print(data.keys())

Fe_arr = data['Fe_arr']
Fd_arr = data['Fd_arr']
p_arr = data['p_arr']
pd_arr = data['pd_arr']
x_tf_arr = data['x_tf_arr']
x_ti_arr = data['x_ti_arr']
Ff_activation_arr = data['Ff_activation_arr']
rho_arr = data['rho_arr']

tank_force = 0.5 * x_tf_arr**2
tank_impedance = 0.5 * x_ti_arr**2

# make time array
N = len(Fe_arr) # number of time steps
t_arr = [i*dt for i in range(N)] 

# Downsample the data by a factor of 10
downsample_factor = 10
Fe_arr = Fe_arr[::downsample_factor]
Fd_arr = Fd_arr[::downsample_factor]

p_arr = p_arr[::downsample_factor]
pd_arr = pd_arr[::downsample_factor]

tank_force = tank_force[::downsample_factor]
tank_impedance = tank_impedance[::downsample_factor]

Ff_activation_arr = Ff_activation_arr[::downsample_factor]
rho_arr = rho_arr[::downsample_factor]

t_arr = t_arr[::downsample_factor]



# plot the force profile 
plt.figure(1)
plt.subplot(311)
plt.plot(t_arr,Fe_arr[:,0])
plt.plot(t_arr,-Fd_arr[:,0])
plt.ylabel('Force x direction')
plt.subplot(312)
plt.plot(t_arr,Fe_arr[:,1])
plt.plot(t_arr,-Fd_arr[:,1])
plt.ylabel('Force y direction')
plt.subplot(313)
plt.plot(t_arr,Fe_arr[:,2])
plt.plot(t_arr,-Fd_arr[:,2])
plt.ylabel('Force z direction')
plt.xlabel('Time (s)')

plt.figure(2)
plt.subplot(311)
plt.plot(t_arr,p_arr[:,0])
plt.plot(t_arr,pd_arr[:,0])
plt.ylabel('x (m)')
plt.subplot(312)
plt.plot(t_arr,p_arr[:,1])
plt.plot(t_arr,pd_arr[:,1])
plt.ylabel('y (m)')
plt.subplot(313)
plt.plot(t_arr,p_arr[:,2])
plt.plot(t_arr,pd_arr[:,2])
plt.ylabel('z (m)')
plt.xlabel('Time (s)')

# plot tank values T_f = 0.5 * x_tf^2, T_i = 0.5 * x_ti^2
plt.figure(3)
plt.subplot(211)
plt.plot(t_arr,tank_force)
plt.ylabel('Force Tank Level')
plt.subplot(212)
plt.plot(t_arr,tank_impedance)
plt.ylabel('Impedance Tank Level')
plt.xlabel('Time (s)')

plt.figure(4)
plt.plot(t_arr,Ff_activation_arr)
plt.ylabel('Activation of Ff')
plt.xlabel('Time (s)')

plt.figure(5)
plt.subplot(311)
plt.plot(t_arr,rho_arr[:,0])
plt.ylabel('Rho Value x')
plt.subplot(312)
plt.plot(t_arr,rho_arr[:,1])
plt.ylabel('Rho Value y')
plt.subplot(313)
plt.plot(t_arr,rho_arr[:,2])
plt.ylabel('Rho Value z')
plt.xlabel('Time (s)')

plt.show()