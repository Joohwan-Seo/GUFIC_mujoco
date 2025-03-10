from gym import Env
from gym import spaces

# import mujoco_py
import mujoco
import mujoco.viewer
import numpy as np
import sympy as sp

from scipy.linalg import expm

import time, csv, os, copy

import pickle

from gym import utils

# import matplotlib.pyplot as plt
from gufic_env.utils.robot_state import RobotState
from gufic_env.utils.mujoco import set_state, set_body_pose_rotm
from gufic_env.utils.misc_func import *



import matplotlib.pyplot as plt

class RobotEnv(Env):
    def __init__(self, robot_name = 'indy7', max_time = 10, show_viewer = False, fz = 10,
                 hole_ori = 'default', testing = False, hole_angle = 0.0, fix_camera = False,
                 tracking = None, gic_only = False, randomized_start = False, inertia_shaping = False
                 ):
        
        self.robot_name = robot_name
        print(self.robot_name)
        self.hole_ori = hole_ori
        self.testing = testing 
        self.tracking = tracking
        self.gic_only = gic_only
        self.randomized_start = randomized_start
        self.inertia_shaping = inertia_shaping

        self.fz = fz
        self.fix_camera = fix_camera
        self.max_time = max_time

        self.hole_angle = hole_angle

        print('==============================================')
        print('USING GEOMETRIC UNIFED FORCE IMPEDANCE CONTROL')
        print('==============================================')

        # self.pd = np.array([0.50, 0.05, 0.13])
        self.pd = np.array([0.40, 0.0, 0.20])
        self.Rd = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        self.pd_default = self.pd
        self.Rd_default = self.Rd
        
        self.p_sphere_center = np.array([0.40, 0.00, 0.0])
        self.R_plate = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        self.z_init_offset = -0.05

        self.initialize_trajectory()

        self.contact_count = 0

        self.show_viewer = show_viewer
        self.load_xml()

        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = self.model.opt.timestep
        self.max_iter = int(max_time/self.dt)

        self.time_step = 0
        self.gd = np.eye(4)

        self.Fe = np.zeros((6,1))
        self.reset()

        ## For the GIC part
        self.Kp = np.eye(3) * np.array([1500, 1500, 500])
        # self.Kp = np.eye(3) * np.array([1000, 1000, 50])

        self.KR = np.eye(3) * np.array([1500, 1500, 1500])

        self.Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        # Velocity-Field Gain
        self.zeta = 5

        self.int_sat = 1000

        if self.tracking is None:
            self.Kp = np.eye(3) * np.array([1500, 1500, 100])
            self.KR = np.eye(3) * np.array([1500, 1500, 1500])
            self.Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            # Default Value
            self.kp_force = 1.0
            self.kd_force = 0.5
            self.ki_force = 8.0

            self.pd = np.array([0.50, 0.00, 0.12])

            # NOTE(JS) Working version of gain for the force tracking, with the 2nd order low-pass filter
            # with cutoff 5hz of the frequency w/o inertia shaping and regular PID control with sat 50
            # self.kp_force = 1.0
            # self.kd_force = 0.5
            # self.ki_force = 4.0

        elif self.tracking == 'circle' or 'line':
            # self.Kp = np.eye(3) * np.array([2500, 2500, 100])
            self.Kp = np.eye(3) * np.array([2000, 2000, 100])
            self.KR = np.eye(3) * np.array([2000, 2000, 2000])
            self.Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            self.kp_force = 2.0
            self.kd_force = 1.0
            self.ki_force = 10.0

            # self.kp_force = 0.3
            # self.kd_force = 0.25
            # self.ki_force = 0.6

            self.pd = np.array([0.50, 0.00, 0.12])

        if self.gic_only == True:
            if self.tracking is None:
                self.Kp = np.eye(3) * np.array([1500, 1500, 800])
            elif self.tracking == 'circle' or 'line':
                self.Kp = np.eye(3) * np.array([2500, 2500, 800])
            self.pd = np.array([0.50, 0.00, 0.12])
            self.pd_default = self.pd


        ## For the force tracking
        self.e_force_prev = np.zeros((6,1))
        self.int_force_prev = np.zeros((6,1))

    
        ## For the energy tank
        self.T_f_low = 0.5
        self.T_f_high = 20
        self.delta_f = 1

        self.T_i_low = 0.5
        self.T_i_high = 100
        self.delta_i = 1

        T_i_init = 90
        T_f_init = 10

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
            model_path = dir + "gufic_env/mujoco_models/Indy7_wiping_sphere.xml"

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
                self.viewer.cam.lookat = np.array([0.5, 0.0, 0.2])  # Center of the scene
                self.viewer.cam.distance = 1.5                     # Distance from the lookat point
                self.viewer.cam.azimuth = 180                       # Horizontal angle in degrees
                self.viewer.cam.elevation = -20                    # Vertical angle in degrees

        else:
            self.viewer = None

    def reset(self, angle_prefix = None):
        _ = self.initial_sample()

        self.iter = 0 
        self.prev_x = np.zeros((3,))
        self.stuck_count = 0
        self.done_count = 0
        
        pd, Rd, vd, wd, dvd, dwd = self.update_desired_trajectory(t=0.0)

        if not self.testing:

            if self.randomized_start:
                rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.05
                rand_rpy = 2*(np.random.rand(3,) - 0.5) * 15 /180 * np.pi
            else:
                rand_xy = np.array([-0.05, 0.05])
                rand_rpy = np.array([15, -15, 15]) * np.pi /180

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            p_init = pd.reshape((-1,1)) + Rd @ np.array([rand_xy[0], rand_xy[1], self.z_init_offset]).reshape(-1,1)
            R_init = Rd @ Rz @ Ry @ Rx

            p_init = p_init.reshape((-1,))
        else:
            if self.randomized_start:
                rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.04
                rand_rpy = 2*(np.random.rand(3,) - 0.5) * 12 /180 * np.pi
            else:
                rand_xy = np.array([0.04, -0.04])
                rand_rpy = np.array([12, -12, 12]) * np.pi /180

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
        self.iter = 0
        self.done_count = 0

        Rt = np.eye(3)
        self.set_hole_pose(self.p_sphere_center, Rt)

        self.robot_state.update()

        p, R = self.robot_state.get_pose()
        self.gd[:3,3] = p
        self.gd[:3,:3] = R

        if self.show_viewer:
            self.viewer.sync()

        print(p_init)
        print(self.pd)

        print('Initialization Complete')
        time.sleep(2)

        return obs

    def initial_sample(self):
        pd = self.pd
        Rd = self.Rd
        

        if self.model.nv == 8:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0, 0])
        elif self.model.nv == 6:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2])

        set_state(self.model, self.data, q0, np.zeros(self.model.nv))
        self.robot_state.update()

        p, R = self.robot_state.get_pose()

        if self.show_viewer:        
            self.viewer.sync()
        ep = R.T @ (p - pd).reshape((-1,1))
        eR = vee_map(Rd.T @ R - R.T @ Rd)

        eg = np.vstack((ep, eR))

        return eg
    
    def initialize_trajectory(self):
        t = sp.symbols('t')
        pd_default_sym = sp.Matrix(self.p_sphere_center)
        Rd_default_sym = sp.Matrix(self.Rd_default)
        total_radian = 1/2*np.pi
        omega_value = total_radian / self.max_time

        if self.tracking is not None:
            theta_y = omega_value * t - total_radian * 0.5


            r_sphere = 0.3
            pd_t_sim = pd_default_sym + sp.Matrix([0, r_sphere * sp.sin(theta_y), -0.1 + r_sphere * sp.cos(theta_y)])
            rotmat_y = sp.Matrix([[sp.cos(-theta_y), 0, sp.sin(-theta_y)], [0, 1, 0], [-sp.sin(-theta_y), 0, sp.cos(-theta_y)]])
            Rd_t_sim = Rd_default_sym @ rotmat_y

        else:
            pd_t_sim = pd_default_sym
            Rd_t_sim = Rd_default_sym

        # Differentiate with symbolic expressions
        dpd_t_sim = sp.diff(pd_t_sim, t)
        dRd_t_sim = sp.diff(Rd_t_sim, t)
        ddpd_t_sim = sp.diff(dpd_t_sim, t)
        ddRd_t_sim = sp.diff(dRd_t_sim, t)

        # Convert symbolic to numpy expressions
        self.pd_t = sp.lambdify(t, pd_t_sim, "numpy")
        self.Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")
        self.dpd_t = sp.lambdify(t, dpd_t_sim, "numpy")
        self.dRd_t = sp.lambdify(t, dRd_t_sim, "numpy")
        self.ddpd_t = sp.lambdify(t, ddpd_t_sim, "numpy")
        self.ddRd_t = sp.lambdify(t, ddRd_t_sim, "numpy")

        return None

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
            # Fd = self.get_force_profile()

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
                if i in [500, 4000, 7999]:
                    print('stopping')

            if i % 1000 == 0:
                print(f"Time Step: {i}")

            if done:
                break

            self.time_step = i

        return p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list, pd_list, Fe_raw_list
    
    def update_desired_trajectory(self, t = None):
        # Return pd, Rd, vd, wd, dvd, dwd
        if t is None:
            t = self.time_step * self.dt
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

        obs = np.zeros((6,1)) # Just putting Dummy variable

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
            fz = 10 * (np.sin(2 * np.pi / 10 * self.time_step * self.dt) + 1)
        
        else:
            fz = self.fz

        Fd = np.array([0, 0, fz, 0, 0, 0])
        return Fd
    
    def get_force_field(self,g, gd):
        eg = self.get_eg(g, gd)

        ep = eg[:3,0]

        # linearly increase force as eg[2] goes to 0 up to 10
        fz_ = self.fz
        if abs(eg[2]) < 0.03:
            fz_ = self.fz

        fz = fz_
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
        Vd_star, dVd_star = self.get_velocity_field(g, Vb.reshape((-1,)), t = self.time_step * self.dt)

        Vd_star = Vd_star.reshape((-1,1))
        dVd_star = dVd_star.reshape((-1,1))

        gd = self.gd
        Rd = gd[:3,:3]
        pd = gd[:3,3]

        g_ed = np.linalg.inv(g) @ gd

        #1 Calculate positional force

        fp = R.T @ Rd @ Kp @ Rd.T @ (p - pd).reshape((-1,1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        #2. calculate PID control for the force tracking
        # Fd = self.get_force_profile().reshape((-1,1))

        # Fd_star = adjoint_g_ed_dual(g_ed) @ Fd
        gd_default = np.eye(4)
        gd_default[:3,:3] = self.Rd_default
        gd_default[:3,3] = self.pd_default
        Fd_star = self.get_force_field(g, gd_default).reshape((-1,1))

        Fe, d_Fe = self.get_FT_value(return_derivative=True)
        Fe = Fe.reshape((-1,1))
        d_Fe = d_Fe.reshape((-1,1))

        # NOTE(JS) Working is version is that to put e_force = - Fe - Fd, with the Fe = -self.robot_state.get_ee_force()
        # Fd should be positive as well

        e_force = -Fe - Fd_star
        de_force = -d_Fe
        int_force = self.int_force_prev + e_force * self.dt


        int_force = np.clip(int_force, -self.int_sat, self.int_sat)

        if self.fz == "time-varying":
            F_f = - self.kp_force * e_force - self.kd_force * de_force - self.ki_force * int_force + Fd_star
        else:
            # integral action with minor loop
            F_f = - self.kp_force * (-Fe) - self.ki_force * int_force - self.kd_force * de_force + Fd_star

        F_f = - self.kp_force * e_force - self.kd_force * de_force - self.ki_force * int_force + Fd_star

        # F_f = np.clip(F_f, -30, 30)


        if self.gic_only == True:
            F_f = np.zeros((6,1))

        #2.5 Apply shaping function to the force control input
        f_d = Fd_star[:3].reshape((-1,))
        m_d = Fd_star[3:].reshape((-1,))

        t = self.time_step * self.dt
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
                # Default
                # if ep[i] >= 0 and np.abs(ep[i]) <= self.d_max:
                #     rho_p[i] = 0.5 * (1 + np.cos(np.pi * ep[i] / self.d_max))
                # elif np.abs(f_f[i]) <= 0.05:
                #     rho_p[i] = 0

                # Try this
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

        # ensure element-wise multiplication
        F_f = F_f * rho

        self.e_force_prev = e_force
        self.int_force_prev = int_force

        # get a scalar value of the inner product of Vb and F_f without any numpy array
        inner_product_f = (Vb.T @ F_f).reshape((-1,))[0]
        # print(inner_product_f)

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


        if self.gic_only:
            ev_mod = Vb - Vd_star
            dVd_star_mod = dVd_star


        # Kd = np.sqrt(np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])) * 10
        Kd = self.Kd

        energy_dissipation = (ev_mod.T @ Kd @ ev_mod)[0,0]
        if energy_dissipation > 10:
            energy_dissipation = 0.1

        ## NOTE Currently, the dissipation term is 0
        # energy_dissipation = 0
        if self.iter % 100 == 0:
            print(f"Sign of impedance inner product:{np.sign(inner_product_i)}, acitvation_impedance: {activation_impedance}")
            print(f"energy_dissipation:{energy_dissipation}" )


        dx_ti = (beta_i / self.x_ti) * (gamma_i * inner_product_i + energy_dissipation) \
                + (alpha_i / self.x_ti) * (1 - gamma_i) * inner_product_i
        
        self.x_ti = self.x_ti + dx_ti * self.dt 

        # GUFIC control law       

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        M_d = np.eye(6) * 10

        # Currently Working version =========================================================
        # tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe) - Fe

        # if self.gic_only:
        #     
        #     tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw) - Fe_raw
        # else:
        #     tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe) - Fe



        Fe_raw = self.get_FT_value_raw().reshape((-1,1))
        if self.inertia_shaping:
            tau_tilde = M_tilde @ (dVd_star_mod + np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw)) - Fe_raw 
        else:
            tau_tilde = M_tilde @ dVd_star_mod -Kd @ ev_mod - fg + F_f_mod
        # tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw) - Fe_raw 

        
        # print('FT Sensor Value:', Fe.reshape((-1,)))

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
    angle = 0
    angle_rad = angle / 180 * np.pi
    randomized_start = False
    inertia_shaping = False
    save = True

    tracking = 'circle'  # None, 'circle', 'line'

    gic_only = False

    assert tracking in [None, 'circle', 'line']

    if tracking is None:
        max_time = 6
    elif tracking == 'line':
        max_time = 8
    elif tracking == 'circle':
        max_time = 10    

    max_time = 8

    RE = RobotEnv(robot_name, show_viewer = show_viewer, max_time = max_time, hole_ori = 'default', 
                  testing = None, fz = 10, 
                  hole_angle = angle_rad, fix_camera = True, tracking = tracking, gic_only = gic_only, 
                  randomized_start=randomized_start, inertia_shaping = inertia_shaping)
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

    ep_list = []

    for i in range(len(p_arr)):
        ep = R_arr[i].T @ (p_arr[i] - pd_arr[i])
        ep_list.append(ep)

    ep = np.asarray(ep_list)


    if tracking is None:
        task_name = 'regulation'
    else:
        task_name = tracking

    if gic_only:
        task_name = task_name + '_gic'
    else:
        task_name = task_name + '_gufic'

    if save:
        with open(f'data/result_{task_name}_IS_{inertia_shaping}_sphere.pkl', 'wb') as f:
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

    plt.figure(6)
    plt.subplot(311)
    plt.plot(ep[:,0])
    plt.ylim([-0.05, 0.05])
    plt.grid()
    plt.ylabel('x (m)')
    plt.subplot(312)
    plt.plot(ep[:,1])
    plt.ylim([-0.05, 0.05])
    plt.grid()
    plt.ylabel('y (m)')
    plt.subplot(313)
    plt.plot(ep[:,2])
    plt.ylabel('z (m)')
    plt.xlabel('Time Step')
    plt.ylim([-0.05, 0.05])
    plt.grid()

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



