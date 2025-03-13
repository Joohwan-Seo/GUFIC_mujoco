from gym import Env
from gym import spaces

# import mujoco_py
import mujoco
import mujoco.viewer
import numpy as np
import sympy as sp

import time, csv, os, copy

import pickle

from gym import utils

# import matplotlib.pyplot as plt
from gufic_env.utils.robot_state import RobotState
from gufic_env.utils.mujoco import set_state, set_body_pose_rotm
from gufic_env.utils.misc_func import *



import matplotlib.pyplot as plt

class RobotEnv:
    def __init__(self, robot_name = 'indy7', max_time = 10, show_viewer = False, fz = 10, observables = None,
                 fix_camera = False, task = "regulation", gic_only = False, randomized_start = False, inertia_shaping = False
                 ):
        
        self.robot_name = robot_name
        self.task = task
        self.gic_only = gic_only
        self.randomized_start = randomized_start
        self.inertia_shaping = inertia_shaping

        if observables is not None:
            self.observables = observables
        else:
            self.observables = ['p', 'pd', 'R', 'Rd']

        self.fz = fz

        self.fix_camera = fix_camera

        print('==============================================')
        print('USING GEOMETRIC IMPEDANCE CONTROL')
        print('==============================================')
        
        self.p_plate = np.array([0.50, 0.00, 0.11])
        self.R_plate = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        if self.task == 'sphere':
            self.p_plate = np.array([0.40, 0.00, 0.0])
        
        self.z_init_offset = -0.1

        self.contact_count = 0

        self.show_viewer = show_viewer
        self.load_xml()

        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = self.model.opt.timestep
        self.max_iter = int(max_time/self.dt)

        self.iter = 0

        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = initialize_trajectory(task = self.task)

        self.Fe = np.zeros((6,1))
        self.reset()

        self.Kp, self.KR, self.Kd = set_gains(controller="GIC", task=self.task)

    def load_xml(self):
        dir = os.getcwd() + '/'
        if self.robot_name == 'ur5e':
            raise NotImplementedError

        elif self.robot_name == 'indy7':
            if self.task == 'sphere':
                model_path = dir + "gufic_env/mujoco_models/Indy7_wiping_sphere.xml"
            else:
                model_path = dir + "gufic_env/mujoco_models/Indy7_wiping.xml"

        elif self.robot_name == 'panda':
            raise NotImplementedError
        
        else:
            raise NotImplementedError

        self.model = mujoco.MjModel.from_xml_path(model_path)
        # self.sim = mujoco.MjSim(self.model)

        # Need to change self.sim with self.data 
        self.data = mujoco.MjData(self.model)
        if self.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

    def reset(self):
        self.iter = 0 

        pd = self.pd_t(0)
        Rd = self.Rd_t(0)

        if self.randomized_start:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.05
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 15 /180 * np.pi
        else:
            rand_xy = np.array([0.05, -0.05])
            rand_rpy = np.array([15, -15, 15]) * np.pi /180

        Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
        Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
        Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

        p_init = pd.reshape((-1,1)) + Rd @ np.array([rand_xy[0], rand_xy[1], self.z_init_offset]).reshape(-1,1)
        R_init = Rd @ Rz @ Ry @ Rx

        p_init = p_init.reshape((-1,))
        
        if self.model.nv == 8:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0, 0])
        elif self.model.nv == 6:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2])

        self.robot_state.gauss_newton_IK(p_init, R_init, q0)

        self.Fe = np.zeros((6,1))

        obs = np.zeros((6,1))

        Rt = np.eye(3)
        self.set_hole_pose(self.p_plate, Rt)

        self.robot_state.update()

        if self.show_viewer:
            self.viewer.sync()

        print('Initialization Complete')
        time.sleep(2)

        return obs

    def run(self):
        p_list = []
        R_list = []
        Fe_list = []
        Fe_raw_list = []
        pd_list = []


        for i in range(self.max_iter):

            pd, Rd, vd, wd, dvd, dwd = self.update_desired_trajectory()

            obs, reward, done, info = self.step()

            p, R = self.robot_state.get_pose()
            Fe = self.get_FT_value()
            Fe_raw = self.get_FT_value_raw()

            p_list.append(p)
            R_list.append(R)
            Fe_list.append(Fe)
            Fe_raw_list.append(Fe_raw)
            pd_list.append(pd)

            # print(reward)

            if self.show_viewer:
                if i % 10 == 0:
                    self.viewer.sync()

            if i % 1000 == 0:
                print(f"Time Step: {i}")

            if done:
                break

        return p_list, R_list, Fe_list, pd_list, Fe_raw_list
    
    def update_desired_trajectory(self):
        # Return pd, Rd, vd, wd, dvd, dwd
        t = self.iter * self.dt
        pd = self.pd_t(t)
        Rd = self.Rd_t(t)

        dpd = self.dpd_t(t)
        dRd = self.dRd_t(t)

        ddpd = self.ddpd_t(t)
        ddRd = self.ddRd_t(t)


        vd = Rd.T @ dpd.reshape((-1,1))
        wd = vee_map(Rd.T @ dRd)

        dvd = Rd.T @ ddpd.reshape((-1,1)) - hat_map(wd) @ Rd.T @ dpd.reshape((-1,1))
        dwd = vee_map(Rd.T @ ddRd - hat_map(wd) @ Rd.T @ dRd)

        return pd.reshape((-1,)), Rd, vd.reshape((-1,)), wd.reshape((-1,)), dvd.reshape((-1,)), dwd.reshape((-1,))

    def step(self):
        self.robot_state.update()

        tau_cmd = self.geometric_impedance_control()

        gripper = 0.03

        self.robot_state.set_control_torque(tau_cmd, gripper)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.sync()

        obs = {}
        p, R = self.robot_state.get_pose()
        pd, Rd, vd, wd, dvd, dwd = self.update_desired_trajectory()
        pd = self.pd_t(self.iter * self.dt).reshape((-1,))
        Rd = self.Rd_t(self.iter * self.dt)
        # Put observables in the obs variable
        for obs_name in self.observables:
            if obs_name == 'p':
                obs['p'] = p
            elif obs_name == 'pd':
                obs['pd'] = pd
            elif obs_name == 'R':
                obs['R'] = R
            elif obs_name == 'Rd':
                obs['Rd'] = Rd
            elif obs_name == 'Fe':
                obs['Fe'] = self.get_FT_value()
            elif obs_name == 'Fe_raw':
                obs['Fe_raw'] = self.get_FT_value_raw()
            elif obs_name == 'Psi':
                obs['Psi'] = 0.5 * np.linalg.norm(p - pd)**2 + np.trace(np.eye(3) - Rd.T @ R)
            else:
                raise ValueError('Invalid observable name')

        if self.iter == self.max_iter -1:
            done = True
        else:
            done = False

        reward = 0
        info = dict()

        self.iter +=1 

        return obs, reward, done, info
    
    def get_FT_value(self, return_derivative = False):
        Fe, dFe = self.robot_state.get_ee_force()
        if return_derivative:
            return -Fe, -dFe
        else:
            return -Fe
        
    def get_FT_value_raw(self):
        Fe, dFe = self.robot_state.get_ee_force_raw()
        return -Fe
    
    def get_eg(self, g, gd):
        p = g[:3,3]
        R = g[:3,:3]

        pd = gd[:3,3]
        Rd = gd[:3,:3]

        ep = R.T @ (p - pd)
        eR = vee_map(Rd.T @ R - R.T @ Rd).reshape((-1,))

        return np.hstack((ep, eR)).reshape((-1,1))
    
    def get_force_profile(self):

        if self.fz == "time-varying":
            fz = 10 * (np.sin(2 * np.pi / 10 * self.iter * self.dt) + 0.5)
        
        else:
            fz = self.fz

        Fd = np.array([0, 0, fz, 0, 0, 0])
        return Fd

    def geometric_impedance_control(self):
        Jb = self.robot_state.get_body_jacobian()

        # M,C,G = self.robot_state.get_dynamic_matrices()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()

        #0 Get impedance gains
        Kp = self.Kp
        KR = self.KR

        p, R = self.robot_state.get_pose()
        # Update trajectory values
        pd, Rd, vd, wd, dvd, dwd = self.update_desired_trajectory()

        g = np.eye(4)
        g[:3,:3] = R
        g[:3,3] = p

        gd = np.eye(4)
        gd[:3,:3] = Rd
        gd[:3,3] = pd

        g_ed = np.linalg.inv(g) @ gd

        Vd = np.hstack((vd, wd)).reshape((-1,1)) # shape of (6,1)
        dVd = np.hstack((dvd, dwd)).reshape((-1,1))
        Vd_star = adjoint_g_ed(g_ed) @ Vd

        dVd_star = adjoint_g_ed_deriv(g, gd, vd, wd, dvd, dwd) @ Vd + adjoint_g_ed(g_ed) @ dVd

        #1 Calculate positional force

        fp = R.T @ Rd @ Kp @ Rd.T @ (p - pd).reshape((-1,1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        Fe, d_Fe = self.get_FT_value(return_derivative=True)
        Fe = Fe.reshape((-1,1))
        d_Fe = d_Fe.reshape((-1,1))

        Vb = self.robot_state.get_body_ee_velocity() #Shape: (6,1)
        Kd = self.Kd

        # GIC control law       

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        M_d = np.eye(6) * 10

        Fe_raw = self.get_FT_value_raw().reshape((-1,1))
        ev = Vb - Vd_star
        if self.inertia_shaping:
            tau_tilde = M_tilde @ (dVd_star + np.linalg.inv(M_d) @ (- Kd @ ev - fg + Fe_raw)) - Fe_raw 
        else:
            tau_tilde = M_tilde @ dVd_star -Kd @ ev - fg
        # tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw) - Fe_raw 

    
        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1,1))

        return tau_cmd.reshape((-1,))
    
    def set_hole_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'hole', pos, R)


if __name__ == "__main__":
    robot_name = 'indy7' 
    show_viewer = True

    randomized_start = False
    inertia_shaping = False
    save = True

    task = 'circle'  # 'regulation', 'circle', 'line'

    assert task in ['regulation', 'circle', 'line', 'sphere']

    if task is None:
        max_time = 6
    elif task == 'line':
        max_time = 8
    elif task == 'circle':
        max_time = 10    

    RE = RobotEnv(robot_name, show_viewer = show_viewer, max_time = max_time, fz = 10, 
                  fix_camera = False, task = task,randomized_start=randomized_start, inertia_shaping = inertia_shaping)
    p_list, R_list, Fe_list, pd_list, Fe_raw_list = RE.run()


    print('Done')

    p_arr = np.asarray(p_list)
    R_arr = np.asarray(R_list)
    # x_tf_arr = np.asarray(x_tf_list)
    # x_ti_arr = np.asarray(x_ti_list)
    Fe_arr = np.asarray(Fe_list)
    # Fd_arr = np.asarray(Fd_list)

    # Ff_arr = np.asarray(Ff_list)
    # Vb_arr = np.asarray(Vb_list)
    # Ff_activation_arr = np.asarray(Ff_activation)
    # rho_arr = np.asarray(rho_list)

    pd_arr = np.asarray(pd_list)

    Fe_raw_arr = np.asarray(Fe_raw_list)


    data = {}
    data['p_arr'] = p_arr
    data['R_arr'] = R_arr
    # data['x_tf_arr'] = x_tf_arr
    # data['x_ti_arr'] = x_ti_arr
    data['Fe_arr'] = Fe_arr
    # data['Fd_arr'] = Fd_arr
    # data['Ff_arr'] = Ff_arr
    # data['Vb_arr'] = Vb_arr
    # data['Ff_activation_arr'] = Ff_activation_arr
    # data['rho_arr'] = rho_arr
    data['pd_arr'] = pd_arr
    data['Fe_raw_arr'] = Fe_raw_arr

    task_name = task


    task_name = task_name + '_gic'

    if save:
        with open(f'data/result_{task_name}_IS_{inertia_shaping}.pkl', 'wb') as f:
            pickle.dump(data, f)

    if show_viewer:
        RE.viewer.close()


    # plot the force profile 
    plt.figure(1)
    plt.plot(-Fe_arr[:,2])
    plt.plot(-Fe_raw_arr[:,2])
    # plt.plot(Fd_arr[:,2])
    plt.legend(['Fe', 'Fe_raw', 'Fd'])
    plt.ylabel('Force z direction')
    plt.xlabel('Time Step')

    plt.figure(2)
    plt.subplot(311)
    plt.plot(p_arr[:,0])
    plt.plot(pd_arr[:,0])
    plt.ylabel('x (m)')
    plt.subplot(312)
    plt.plot(p_arr[:,1])
    plt.plot(pd_arr[:,1])
    plt.ylabel('y (m)')
    plt.subplot(313)
    plt.plot(p_arr[:,2])
    plt.plot(pd_arr[:,2])
    plt.ylabel('z (m)')
    plt.xlabel('Time Step')

    plt.show()




