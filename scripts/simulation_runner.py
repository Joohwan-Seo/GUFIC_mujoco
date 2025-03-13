from gufic_env.env_gufic_velocity_field import RobotEnv as GUFICEnv
from gufic_env.env_gic_trajectory_tracking import RobotEnv as GICEnv

import pickle
import numpy as np

def main(control_method = 'gufic', task = 'circle', show_viewer = False):
    robot_name = 'indy7' 
    randomized_start = False
    inertia_shaping = False
    save = True

    assert control_method in ['gufic', 'gic']
    assert task in ['regulation', 'circle', 'line', 'sphere']


    if task == 'regulation':
        max_time = 6
    elif task == 'line':
        max_time = 8
    elif task == 'circle' or task == 'sphere':
        max_time = 10

    if control_method == 'gufic':
        observables = ['p', 'pd', 'R', 'Rd', 'x_tf', 'x_ti', 'Fe', 'Fd', 'rho', 'Fe_raw', 'Psi']
        agent = GUFICEnv(robot_name, show_viewer = show_viewer, max_time = max_time, fz = 10, observables = observables,
                        fix_camera = False, task = task, randomized_start=randomized_start, inertia_shaping = inertia_shaping)
    elif control_method == 'gic':
        observables = ['p', 'pd', 'R', 'Rd', 'Fe', 'Fe_raw', 'Psi']
        agent = GICEnv(robot_name, show_viewer = show_viewer, max_time = max_time, fz = 10, observables = observables,
                    fix_camera = False, task = task, randomized_start=randomized_start, inertia_shaping = inertia_shaping)
    else:
        raise ValueError('Invalid control method')

    dt = agent.dt
    max_iter = agent.max_iter

    # Set observables as a dictionary
    data = {}
    for val in observables:
        data[val] = []

    # Run the simulation
    for i in range(max_iter):
        obs, reward, done, info = agent.step()

        for val in observables:
            data[val].append(obs[val])

        if show_viewer:
            if i % 10 == 0:
                agent.viewer.sync()
        pass

    if show_viewer:
        agent.viewer.close()

    # Convert lists to the numpy arrays
    for val in observables:
        data[val] = np.asarray(data[val])
    
    data['dt'] = dt

    # Save the data
    file_name = f"data/result_{task}_{control_method}_IS_{inertia_shaping}.pkl"

    if save:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data saved to {file_name}")

if __name__ == '__main__':
    show_viewer = False
    control_methods = ['gufic', 'gic']
    tasks = ['regulation', 'circle', 'line', 'sphere']

    # tasks = ['sphere']

    for control_method in control_methods:
        for task in tasks:
            main(control_method, task, show_viewer)
            print(f"Finished {control_method} {task}")