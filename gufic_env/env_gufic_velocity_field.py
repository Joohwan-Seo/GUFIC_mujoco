# import mujoco
import mujoco
import mujoco.viewer
import numpy as np
import sympy as sp

from scipy.linalg import expm

import time, csv, os, copy

import pickle

# import matplotlib.pyplot as plt
from gufic_env.utils.robot_state import RobotState
from gufic_env.utils.mujoco import set_state, set_body_pose_rotm
from gufic_env.utils.misc_func import *



import matplotlib.pyplot as plt

class RobotEnv:
    def __init__(self, robot_name = 'indy7', max_time = 10, show_viewer = False, fz = 10, observables = None,
                 fix_camera = False, task = 'regulation', randomized_start = False, inertia_shaping = False
                 ):
        
        self.robot_name = robot_name
        self.task = task
        self.randomized_start = randomized_start
        self.inertia_shaping = inertia_shaping

        if observables is not None:
            self.observables = observables
        else:
            self.observables = ['p', 'pd', 'R', 'Rd', 'x_tf', 'x_ti', 'Fe', 'Fe_raw', 'Fd', 'rho']

        self.fz = fz
        self.fix_camera = fix_camera

        print('==============================================')
        print('USING GEOMETRIC UNIFED FORCE IMPEDANCE CONTROL')
        print('==============================================')

        self.p_plate = np.array([0.50, 0.00, 0.11])
        self.R_plate = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        if self.task == 'sphere':
            self.p_plate = np.array([0.40, 0.00, 0.0])
        
        self.z_init_offset = -0.1

        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = initialize_trajectory(task = self.task)

        self.show_viewer = show_viewer
        self.load_xml()

        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = self.model.opt.timestep
        self.max_iter = int(max_time/self.dt)

        self.iter = 0

        self.Fe = np.zeros((6,1))
        self.reset()

        self.Kp, self.KR, self.Kd, self.kp_force, self.kd_force, self.ki_force, self.zeta = set_gains(controller = 'GUFIC', task = self.task)

        # print("Gains:", self.Kp, self.KR, self.Kd, self.kp_force, self.kd_force, self.ki_force, self.zeta)
        # print(self.pd_t(0))

        self.int_sat = 50

        ## For the force tracking
        self.e_force_prev = np.zeros((6,1))
        self.int_force_prev = np.zeros((6,1))

        ## For the energy tank
        self.T_f_low = 0.5
        self.T_f_high = 20
        self.delta_f = 1

        self.T_i_low = 0.5
        self.T_i_high = 20
        self.delta_i = 1

        T_i_init = 10
        T_f_init = 10

        if self.task == 'sphere':
            T_i_init = 90
            self.T_i_high = 100

        self.x_tf = np.sqrt(2 * T_f_init)
        self.x_ti = np.sqrt(2 * T_i_init)

        self.T_f = 0.5 * self.x_tf**2
        self.T_i = 0.5 * self.x_ti**2

        self.d_max = 0.03
        self.eR_norm_max = 0.05

        ####### Dummy for the printing
        self.Ff_list = []
        self.Vb_list = []
        self.Ff_activation = []
        self.rho_list = []
        self.Fd_star_list = []
        self.Fi_activation = []

    def load_xml(self):
        # dir = "/home/joohwan/deeprl/research/GIC_Learning_public/"
        dir = os.getcwd() + '/'
        if self.robot_name == 'ur5e':
            raise NotImplementedError

        elif self.robot_name == 'indy7':
            if self.task == "sphere":
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
            if self.fix_camera:
                self.viewer.cam.fixedcamid = 0      # Use a predefined camera from your XML (if available)
                self.viewer.cam.trackbodyid = -1      # Disable tracking any body
                # Alternatively, if you want to set a free camera pose manually:
                self.viewer.cam.lookat = np.array([0.5, 0.0, 0.3])  # Center of the scene
                self.viewer.cam.distance = 1.5                     # Distance from the lookat point
                self.viewer.cam.azimuth = 180                       # Horizontal angle in degrees
                self.viewer.cam.elevation = -20                    # Vertical angle in degrees
        else:
            self.viewer = None

    def reset(self, angle_prefix = None):
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

        p, R = self.robot_state.get_pose()

        # Reset integration variable gd
        self.gd = np.eye(4)
        self.gd[:3,3] = p
        self.gd[:3,:3] = R

        if self.show_viewer:
            self.viewer.sync()

        print('Initialization Complete')
        time.sleep(2)

        return obs

    def run(self):
        p_list = []
        R_list = []
        x_tf_list = []
        x_ti_list = []
        Fe_list = []
        Fd_list = []

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
            x_tf_list.append(self.x_tf)
            x_ti_list.append(self.x_ti)
            Fe_list.append(Fe)
            Fe_raw_list.append(Fe_raw)
            Fd_list.append(0)
            pd_list.append(pd)

            # print(reward)

            if self.show_viewer:
                if i % 10 == 0:
                    self.viewer.sync()
                if i in [4000]:
                    # print('Stopping here')
                    pass

            if i % 1000 == 0:
                print(f"Time Step: {i}")

            if done:
                break

            # self.iter = i

        return p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list, pd_list, Fe_raw_list
    
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
    
    def get_velocity_field(self, g, V, t):
        zeta = self.zeta
        pd = self.pd_t(t).reshape((-1,))
        Rd = self.Rd_t(t)

        dpd = self.dpd_t(t).reshape((-1,))
        dRd = self.dRd_t(t)

        ddpd = self.ddpd_t(t).reshape((-1,))
        ddRd = self.ddRd_t(t)

        p = g[:3,3]
        R = g[:3,:3]

        v = V[:3] 
        w = V[3:]

        Vd_star = np.zeros(6,)
        vd_star = R.T @ dRd @ Rd.T @ (p - pd) + R.T @ dpd - zeta * R.T @ (p - pd)
        wd_star = vee_map(R.T @ dRd @ Rd.T @ R - zeta * (Rd.T @ R - R.T @ Rd)).reshape((-1,))

        Vd_star[:3] = vd_star
        Vd_star[3:] = wd_star

        dVd_star = np.zeros(6,)
        term1 = -hat_map(w) @ R.T @ dRd @ Rd.T @ R + R.T @ ddRd @ Rd.T @ R + R.T @ dRd @ dRd.T @ R + R.T @ dRd @ Rd.T @ R @ hat_map(w)
        term2 = -hat_map(w) @ R.T @ dRd @ Rd.T @ (p - pd) + R.T @ ddRd @ Rd.T @ (p - pd) + R.T @ dRd @ dRd.T @ (p - pd) \
                + R.T @ dRd @ Rd.T @ (R.T @ v - pd) - hat_map(w) @ R.T @ dpd + R.T @ ddpd
        term3 = dRd.T @ R + Rd.T @ R @ hat_map(w) + hat_map(w) @ R.T @ Rd - R.T @ dRd
        term4 = - hat_map(w) @ R.T @ (p - pd) + v - R.T @ dpd
        dvd_star = term2 - zeta * term4
        dwd_star = vee_map(term1 - zeta * term3).reshape((-1,))

        dVd_star[:3] = dvd_star
        dVd_star[3:] = dwd_star

        return Vd_star, dVd_star


    def step(self):
        self.robot_state.update()

        tau_cmd = self.geometric_unified_force_impedance_control()

        gripper = 0.03

        self.robot_state.set_control_torque(tau_cmd, gripper)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.sync()

        obs = {}
        # Put observables in the obs variable
        p, R = self.robot_state.get_pose()

        pd = self.pd_t(self.iter * self.dt).reshape((-1,))
        Rd = self.Rd_t(self.iter * self.dt)

        for obs_name in self.observables:
            if obs_name == 'p':
                obs['p'] = p
            elif obs_name == 'pd':
                obs['pd'] = pd
            elif obs_name == 'R':
                obs['R'] = R
            elif obs_name == 'Rd':
                obs['Rd'] = Rd
            elif obs_name == 'x_tf':
                obs['x_tf'] = self.x_tf
            elif obs_name == 'x_ti':
                obs['x_ti'] = self.x_ti
            elif obs_name == 'Fe':
                obs['Fe'] = self.get_FT_value()
            elif obs_name == 'Fe_raw':
                obs['Fe_raw'] = self.get_FT_value_raw()
            elif obs_name == 'Fd':
                obs['Fd'] = self.get_force_field(self.gd, self.gd)
            elif obs_name == 'rho':
                obs['rho'] = self.rho
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
    
    def get_force_field(self,g, gd):
        fz = self.fz

        Fd = np.array([0, 0, fz, 0, 0, 0])
        return Fd


    def geometric_unified_force_impedance_control(self):
        Jb = self.robot_state.get_body_jacobian()

        # M,C,G = self.robot_state.get_dynamic_matrices()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()

        #0 Get impedance gains
        Kp = self.Kp
        KR = self.KR

        p, R = self.robot_state.get_pose()

        g = np.eye(4)
        g[:3,:3] = R
        g[:3,3] = p

        Vb = self.robot_state.get_body_ee_velocity() # Shape: (6,1)

        # Update trajectory values
        Vd_star, dVd_star = self.get_velocity_field(g, Vb.reshape((-1,)), t = self.iter * self.dt)

        Vd_star = Vd_star.reshape((-1,1))
        dVd_star = dVd_star.reshape((-1,1))

        #Original GIC Law placeholder
        gd = self.gd
        Rd = gd[:3,:3]
        pd = gd[:3,3]

        g_ed = np.linalg.inv(g) @ gd

        #1 Calculate positional force
        fp = R.T @ Rd @ Kp @ Rd.T @ (p - pd).reshape((-1,1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        gd_bar = np.eye(4)
        t = self.iter * self.dt
        gd_bar[:3,:3] = self.Rd_t(t)
        gd_bar[:3,3] = self.pd_t(t).reshape((-1,))
        Fd_star = self.get_force_field(g, gd_bar).reshape((-1,1))

        Fe, d_Fe = self.get_FT_value(return_derivative=True)
        Fe = Fe.reshape((-1,1))
        d_Fe = d_Fe.reshape((-1,1))

        # NOTE(JS) Working is version is that to put e_force = - Fe - Fd, with the Fe = -self.robot_state.get_ee_force()
        # Fd should be positive as well

        e_force = -Fe - Fd_star
        de_force = -d_Fe
        int_force = self.int_force_prev + e_force * self.dt


        int_force = np.clip(int_force, -self.int_sat, self.int_sat)

        if self.fz == "time-varying": # Regular PID Control
            F_f = - self.kp_force * e_force - self.kd_force * de_force - self.ki_force * int_force + Fd_star
        else: # Integral action with minor loop
            F_f = - self.kp_force * (-Fe) - self.ki_force * int_force - self.kd_force * de_force + Fd_star

        #2.5 Apply shaping function to the force control input
        f_d = Fd_star[:3].reshape((-1,))
        m_d = Fd_star[3:].reshape((-1,))

        t = self.iter * self.dt
        gd_t = np.eye(4)
        gd_t[:3,:3] = self.Rd_t(t)
        gd_t[:3,3] = self.pd_t(t).reshape((-1,))
        eg = self.get_eg(g, gd_t)

        ep = eg[:3,0]
        eR = eg[3:,0]

        rho_p = np.zeros((3,))
        rho_R = np.zeros((3,))

        if ep @ f_d <= 0:
            rho_p[:3] = 1
        elif ep @ f_d > 0:
            for i in range(3):
                if np.abs(ep[i]) <= self.d_max:
                    rho_p[i] = 0.5 * (1 + np.cos(np.pi * ep[i] / self.d_max))
                elif np.abs(f_d[i]) <= 0.05:
                    rho_p[i] = 0
        else:
            rho_p[:3] = 0

        eR_norm = np.linalg.norm(eR)
        if eR @ m_d <= 0:
            rho_R[:3] = 1
        elif eR @ m_d > 0:
            if eR_norm >= self.eR_norm_max:
                rho_R[:3] = 0.5 * (1 + np.cos(np.pi * eR_norm / self.eR_norm_max))
        else:
            rho_R[:3] = 0

        rho = np.block([rho_p, rho_R]).reshape((-1,1))
        self.rho = rho

        # ensure element-wise multiplication
        F_f = F_f * rho

        self.e_force_prev = e_force
        self.int_force_prev = int_force

        # get a scalar value of the inner product of Vb and F_f without any numpy array
        inner_product_f = (Vb.T @ F_f).reshape((-1,))[0]

        self.T_f = 0.5 * self.x_tf**2

        if inner_product_f < 0:
            gamma_f = 1
        else:
            gamma_f = 0

        if self.T_f <= self.T_f_high:
            beta_f = 1
        else:
            beta_f = 0

        if self.T_f >= self.T_f_low + self.delta_f:
            alpha_f = 1
        elif self.T_f <= self.T_f_low + self.delta_f and self.T_f >= self.T_f_low:
            alpha_f = 0.5 * (1 - np.cos(np.pi * (self.T_f - self.T_f_low) / self.delta_f))
        elif self.T_f < self.T_f_low:
            alpha_f = 0
        
        dx_tf = - (beta_f / self.x_tf) * gamma_f * inner_product_f + (alpha_f / self.x_tf) * (gamma_f -1) * inner_product_f
        self.x_tf = self.x_tf + dx_tf * self.dt
        self.T_f = 0.5 * self.x_tf**2

        activation_force = gamma_f + alpha_f * (1 - gamma_f)
        F_f_mod = activation_force * F_f

        #4. Modified Impedance Control
        inner_product_i = (Vd_star.T @ (F_f_mod + Fe)).reshape((-1,))[0]

        self.T_i = 0.5 * self.x_ti**2

        if inner_product_i > 0:
            gamma_i = 1
        else:
            gamma_i = 0
        
        if self.T_i <= self.T_i_high:
            beta_i = 1
        elif self.T_i > self.T_i_high:
            beta_i = 0

        if self.T_i >= self.T_i_low + self.delta_i:
            alpha_i = 1
        elif self.T_i <= self.T_i_low + self.delta_i and self.T_i >= self.T_i_low:
            alpha_i = 0.5 * (1 - np.cos(np.pi * (self.T_i - self.T_i_low) / self.delta_i))
        else:
            alpha_i = 0

        activation_impedance = gamma_i + alpha_i * (1 - gamma_i)
        Vd_star_mod = activation_impedance * Vd_star
        dVd_star_mod = activation_impedance * dVd_star
        ev_mod = Vb - Vd_star_mod

        # calculate next_step gd
        Vd_mod = adjoint_g_ed(np.linalg.inv(g_ed)) @ Vd_star_mod
        Vd_mod_hat = np.zeros((4,4))
        Vd_mod_hat[:3,:3] = hat_map(Vd_mod[3:,0])
        Vd_mod_hat[:3,3] = Vd_mod[:3,0]
        self.gd = gd @ expm(Vd_mod_hat * self.dt)

        Kd = self.Kd

        energy_dissipation = (ev_mod.T @ Kd @ ev_mod)[0,0]
        if energy_dissipation > 10:
            energy_dissipation = 0.1

        if self.iter % 100 == 0: #NOTE(JS) For the Debugging

            # print(f"Sign of impedance inner product:{np.sign(inner_product_i)}, acitvation_impedance: {activation_impedance}")
            # print(f"energy_dissipation:{energy_dissipation}" )
            pass


        dx_ti = (beta_i / self.x_ti) * (gamma_i * inner_product_i + energy_dissipation) \
                + (alpha_i / self.x_ti) * (1 - gamma_i) * inner_product_i
        
        self.x_ti = self.x_ti + dx_ti * self.dt 

        # GUFIC control law       

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        M_d = np.eye(6) * 10

        Fe_raw = self.get_FT_value_raw().reshape((-1,1))
        if self.inertia_shaping:
            tau_tilde = M_tilde @ (dVd_star_mod + np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw)) - Fe_raw 
        else:
            tau_tilde = M_tilde @ dVd_star_mod -Kd @ ev_mod - fg + F_f_mod

        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1,1))

        ####### Save all the dummy variables
        self.Fd_star_list.append(Fd_star)
        self.Ff_list.append(F_f)
        self.Vb_list.append(Vb)
        self.Ff_activation.append(activation_force)
        self.Fi_activation.append(activation_impedance)
        self.rho_list.append(rho)

        return tau_cmd.reshape((-1,))
    
    def set_hole_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'hole', pos, R)


if __name__ == "__main__":
    robot_name = 'indy7' 
    show_viewer = True
    randomized_start = False
    inertia_shaping = False
    save = True

    task = 'sphere'  # "regulation", 'circle', 'line'

    assert task in ['regulation', 'circle', 'line', 'sphere']

    if task is None:
        max_time = 6
    elif task == 'line':
        max_time = 8
    elif task == 'circle':
        max_time = 10
    elif task == 'sphere':
        max_time = 10

    RE = RobotEnv(robot_name, show_viewer = show_viewer, max_time = max_time, fz = 10, 
                  fix_camera = True, task = task, randomized_start=randomized_start, inertia_shaping = inertia_shaping)
    p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list, pd_list, Fe_raw_list = RE.run()

    Ff_list = RE.Ff_list
    Vb_list = RE.Vb_list
    Ff_activation = RE.Ff_activation
    rho_list = RE.rho_list
    Fd_list = RE.Fd_star_list


    print('Done')

    p_arr = np.asarray(p_list)
    R_arr = np.asarray(R_list)
    x_tf_arr = np.asarray(x_tf_list)
    x_ti_arr = np.asarray(x_ti_list)
    Fe_arr = np.asarray(Fe_list)
    Fd_arr = np.asarray(Fd_list)

    Ff_arr = np.asarray(Ff_list)
    Vb_arr = np.asarray(Vb_list)
    Ff_activation_arr = np.asarray(Ff_activation)
    rho_arr = np.asarray(rho_list)

    pd_arr = np.asarray(pd_list)

    Fe_raw_arr = np.asarray(Fe_raw_list)

    # Perform the inner_product_f value
    # inner_product_f_arr = np.zeros((len(Vb_arr),1))
    # for i in range(len(Vb_arr)):
    #     inner_product_f_arr[i] = (Vb_arr[i].T @ Ff_arr[i]).reshape((-1,))[0]

    data = {}
    data['p_arr'] = p_arr
    data['R_arr'] = R_arr
    data['x_tf_arr'] = x_tf_arr
    data['x_ti_arr'] = x_ti_arr
    data['Fe_arr'] = Fe_arr
    data['Fd_arr'] = Fd_arr
    data['Ff_arr'] = Ff_arr
    data['Vb_arr'] = Vb_arr
    data['Ff_activation_arr'] = Ff_activation_arr
    data['rho_arr'] = rho_arr
    data['pd_arr'] = pd_arr
    data['Fe_raw_arr'] = Fe_raw_arr

    task_name = task

    task_name = task_name + '_gufic'

    if save:
        with open(f'data/result_{task_name}_IS_{inertia_shaping}.pkl', 'wb') as f:
            pickle.dump(data, f)

    if show_viewer:
        RE.viewer.close()


    # plot the force profile 
    plt.figure(1)
    plt.plot(-Fe_arr[:,2])
    plt.plot(-Fe_raw_arr[:,2])
    plt.plot(Fd_arr[:,2])
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

    # plot tank values T_f = 0.5 * x_tf^2, T_i = 0.5 * x_ti^2
    plt.figure(3)
    plt.subplot(211)
    plt.plot(0.5 * x_tf_arr**2)
    plt.ylabel('Force Tank Level')
    plt.subplot(212)
    plt.plot(0.5 * x_ti_arr**2)
    plt.ylabel('Impedance Tank Level')
    plt.xlabel('Time Step')

    plt.figure(4)
    plt.plot(Ff_activation_arr)
    plt.ylabel('Activation of Ff')

    plt.figure(5)
    plt.plot(rho_arr[:,2])
    plt.ylabel('Rho Value')

    plt.show()




