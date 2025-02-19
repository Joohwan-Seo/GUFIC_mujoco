from gym import Env
from gym import spaces

# import mujoco_py
import mujoco
import mujoco.viewer
import numpy as np
import time, csv, os, copy

from gym import utils

# import matplotlib.pyplot as plt
from gufic_env.utils.robot_state import RobotState
from gufic_env.utils.mujoco import set_state, set_body_pose_rotm
from gufic_env.utils.misc_func import vee_map, hat_map, rotmat_x

import matplotlib.pyplot as plt



class RobotEnv:
    def __init__(self, robot_name = 'indy7', max_time = 10, show_viewer = False, 
                 obs_type = 'pos', hole_ori = 'default', testing = False, reward_version = None, window_size = 1, 
                 use_ext_force = False, act_type = 'default', mixed_obs = False, hole_angle = 0.0, in_dist = True, fix_camera = False,
                 tracking = None):
        
        self.robot_name = robot_name
        print(self.robot_name)
        self.hole_ori = hole_ori
        self.testing = testing 
        self.tracking = tracking


        self.reward_version = reward_version
        self.window_size = window_size
        self.use_external_force = use_ext_force
        self.act_type = act_type
        self.in_dist = in_dist
        self.fix_camera = fix_camera

        self.hole_angle = hole_angle

        if mixed_obs: # To obtain the data for GIC + CEV case
            self.obs_is_Cart = True
        else:
            self.obs_is_Cart = False
        print('=================================')
        print('USING GEOMETRIC IMPEDANCE CONTROL')
        print('=================================')



        #NOTE(JS) The determinant of the desired rotation matrix should be always 1.
        # (by the definition of the rotational matrix.)

        # self.pd = np.array([0.50, 0.05, 0.13])
        self.pd = np.array([0.50, 0.00, 0.13])
        self.Rd = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        self.pd_default = self.pd
        self.Rd_default = self.Rd
        
        self.p_plate = np.array([0.50, 0.00, 0.11])
        self.R_plate = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        self.z_init_offset = -0.1

        self.show_viewer = show_viewer
        self.load_xml()

        self.obs_type = obs_type
        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = 0.001
        self.max_iter = int(max_time/self.dt)

        self.time_step = 0

        self.Fe = np.zeros((6,1))
        self.reset()

        ## For the GIC part
        self.Kp = np.eye(3) * np.array([1500, 1500, 100])
        # self.Kp = np.eye(3) * np.array([1000, 1000, 50])

        self.KR = np.eye(3) * np.array([1500, 1500, 1500])

        self.Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        ## For the force tracking
        self.e_force_prev = np.zeros((6,1))
        self.int_force_prev = np.zeros((6,1))

        ######## Working version of gain for the force tracking #0109
        # self.kp_force = 0.2
        # self.kd_force = 0
        # self.ki_force = 0.6

        self.kp_force = 1.0
        self.kd_force = 0.25
        self.ki_force = 1.5

        ## For the energy tank
        self.T_f_low = 0.01
        self.T_f_high = 20
        self.delta_f = 1

        self.T_i_low = 0.01
        self.T_i_high = 20
        self.delta_i = 1

        self.x_tf = np.sqrt(2 * 10)
        self.x_ti = np.sqrt(2 * 10)

        self.T_f = 0.5 * self.x_tf**2
        self.T_i = 0.5 * self.x_ti**2

        self.d_max = 0.04
        self.eR_norm_max = 0.05

        ####### Dummy for the printing
        self.Ff_list = []
        self.Vb_list = []
        self.Ff_activation = []
        self.rho_list = []

    def load_xml(self):
        # dir = "/home/joohwan/deeprl/research/GIC_Learning_public/"
        dir = os.getcwd() + '/'
        if self.robot_name == 'ur5e':
            raise NotImplementedError

        elif self.robot_name == 'indy7':
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

    def reset(self, angle_prefix = None):
        _ = self.initial_sample()

        self.iter = 0 
        self.prev_x = np.zeros((3,))
        self.stuck_count = 0
        self.done_count = 0

        if not self.testing:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.05
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 15 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            p_init = self.pd.reshape((-1,1)) + self.Rd @ np.array([rand_xy[0], rand_xy[1], self.z_init_offset]).reshape(-1,1)
            R_init = self.Rd @ Rz @ Ry @ Rx

            p_init = p_init.reshape((-1,))
        else:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.04
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 12 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            p_init = self.pd.reshape((-1,1)) + self.Rd @ np.array([rand_xy[0], rand_xy[1], self.z_init_offset]).reshape(-1,1)
            R_init = self.Rd @ Rz @ Ry @ Rx

            p_init = p_init.reshape((-1,))

        
        if self.model.nv == 8:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0, 0])
        elif self.model.nv == 6:
            q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2])

        self.robot_state.gauss_newton_IK(p_init, R_init, q0)

        self.Fe = np.zeros((6,1))

        obs = self._get_obs()

        self.iter = 0
        self.done_count = 0

        Rt = np.eye(3)
        self.set_hole_pose(self.p_plate, Rt)

        self.robot_state.update()

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

    def run(self):
        p_list = []
        R_list = []
        x_tf_list = []
        x_ti_list = []
        Fe_list = []
        Fd_list = []


        for i in range(self.max_iter):

            self.pd, self.Rd = self.update_desired_trajectory()

            obs, reward, done, info = self.step()

            p, R = self.robot_state.get_pose()
            Fe = self.get_FT_value()
            Fd = self.get_force_profile()

            p_list.append(p)
            R_list.append(R)
            x_tf_list.append(self.x_tf)
            x_ti_list.append(self.x_ti)
            Fe_list.append(Fe)
            Fd_list.append(Fd)

            # print(reward)

            if self.show_viewer:
                if i % 10 == 0:
                    self.viewer.sync()

            if i % 1000 == 0:
                print(f"Time Step: {i}")

            if done:
                break

            self.time_step = i

        return p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list
    
    def update_desired_trajectory(self):

        if self.tracking is not None:
            if self.tracking == "circle":
                t = self.time_step * self.dt
                r = 0.1

                pd = self.pd_default + r * np.array([np.cos(t * 0.5), -np.sin(t * 0.5), 0])
                Rd = self.Rd_default
            elif self.tracking == "line":
                t = self.time_step * self.dt
                pd = self.pd_default + np.array([0, 0.05 * t, 0])
                Rd = self.Rd_default
        else:
            pd = self.pd
            Rd = self.Rd

        return pd, Rd

    def step(self):
        self.robot_state.update()

        tau_cmd = self.geometric_unified_force_impedance_control()

        # if self.testing:
            # print(action)

        gripper = 0.03

        self.robot_state.set_control_torque(tau_cmd, gripper)

        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.sync()

        obs = self._get_obs()

        if self.iter == self.max_iter -1:
            done = True
        else:
            done = False

        reward = 0
        info = dict()

        self.iter +=1 

        return obs, reward, done, info
    
    def _get_obs(self):
        eg = self.get_eg()
        eV = self.get_eV()
        Fe = self.get_FT_value()

        if self.obs_type == 'pos_vel':
            raw_obs = np.vstack((eg,eV)).reshape((-1,))
        elif self.obs_type == 'pos':
            raw_obs = eg.reshape((-1,)) 

        if self.window_size == 1:
            obs = raw_obs
        else:
            self.memorize(raw_obs)
            obs = np.asarray(self.obs_memory).reshape((-1,))

        if self.obs_is_Cart:
            x, R = self.robot_state.get_pose()
            ep = (x - self.pd).reshape((-1,1))

            Rd1 = self.Rd[:,0]; Rd2 = self.Rd[:,1]; Rd3 = self.Rd[:,2]
            R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

            eR = -((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

            eg = np.vstack((ep,eR))
            obs = eg.reshape((-1,))

        return obs

    
    def memorize(self,obs):
        _temp = copy.deepcopy(self.obs_memory)
        for i in range(self.window_size):
            if i < self.window_size - 1:
                self.obs_memory[i+1] = _temp[i]

        self.obs_memory[0] = obs

    def get_eg(self):
        x, R = self.robot_state.get_pose()
        ep = R.T @ (x - self.pd).reshape((-1,1))
        eR = vee_map(self.Rd.T @ R - R.T @ self.Rd)

        eg = np.vstack((ep,eR))

        return eg

    def get_eV(self):
        return self.robot_state.get_body_ee_velocity()
    
    def get_FT_value(self, return_derivative = False):
        Fe, dFe = self.robot_state.get_ee_force()
        if return_derivative:
            return -Fe, -dFe
        else:
            return -Fe
    
    def get_force_profile(self):
        Fd = np.array([0, 0, 20, 0, 0, 0])
        return Fd

    def geometric_unified_force_impedance_control(self):
        Jb = self.robot_state.get_body_jacobian()

        # M,C,G = self.robot_state.get_dynamic_matrices()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()

        #0 Get impedance gains
        Kp = self.Kp
        KR = self.KR

        #1 Calculate positional force
        x, R = self.robot_state.get_pose()
        xd, Rd = self.pd, self.Rd

        fp = R.T @ Rd @ Kp @ Rd.T @ (x - xd).reshape((-1,1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        #2. calculate PID control for the force tracking
        Fd = self.get_force_profile().reshape((-1,1))
        Fe, d_Fe = self.get_FT_value(return_derivative=True)
        Fe = Fe.reshape((-1,1))
        d_Fe = d_Fe.reshape((-1,1))

        # NOTE(JS) Working is thate to put e_force = - Fe - Fd, with the Fe = -self.robot_state.get_ee_force()
        # Fd should be positive as well

        e_force = -Fe - Fd
        de_force = -d_Fe
        int_force = self.int_force_prev + e_force * self.dt

        F_f = - self.kp_force * e_force - self.kd_force * de_force - self.ki_force * int_force + Fd

        #2.5 Apply shaping function to the force control input
        f_f = Fd[:3].reshape((-1,))
        m_f = Fd[3:].reshape((-1,))

        eg = self.get_eg()

        ep = eg[:3,0]
        eR = eg[3:,0]

        rho_p = np.zeros((3,))
        rho_R = np.zeros((3,))

        if self.time_step % 100 == 0:
            print(ep[2], f_f[2])
            print(ep @ f_f)

        if ep @ f_f <= 0:
            rho_p[:3] = 1
        elif ep @ f_f > 0:
            for i in range(3):
                if ep[i] >= 0 and np.abs(ep[i]) <= self.d_max:
                    rho_p[i] = 0.5 * (1 + np.cos(np.pi * ep[i] / self.d_max))
                elif np.abs(f_f[i]) <= 0.05:
                    rho_p[i] = 0
        else:
            rho_p[:3] = 0

        eR_norm = np.linalg.norm(eR)
        if eR @ m_f <= 0:
            rho_R[:3] = 1
        elif eR @ m_f > 0:
            if eR_norm >= self.eR_norm_max:
                rho_R[:3] = 0.5 * (1 + np.cos(np.pi * eR_norm / self.eR_norm_max))
        else:
            rho_R[:3] = 0

        rho = np.block([rho_p, rho_R]).reshape((-1,1))

        # ensure element-wise multiplication
        F_f = F_f * rho

        self.e_force_prev = e_force
        self.int_force_prev = int_force

        #3 update the energy tank state
        Vb = self.robot_state.get_body_ee_velocity() #Shape: (6,1)

        # get a scalar value of the inner product of Vb and F_f without any numpy array
        inner_product_f = (Vb.T @ F_f).reshape((-1,))[0]
        # print(inner_product_f)

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
        else:
            alpha_f = 0
        
        dx_tf = - (beta_f / self.x_tf) * gamma_f * inner_product_f + (alpha_f / self.x_tf) * (gamma_f -1) * inner_product_f
        self.x_tf = self.x_tf + dx_tf * self.dt
        self.T_f = 0.5 * self.x_tf**2

        F_f_mod = (gamma_f + alpha_f * (1 - gamma_f)) * F_f

        #4. Modified Impedance Control
        Vd_star = np.zeros((6,1)) # To be updated later

        inner_product_i = (Vb.T @ (F_f_mod + Fe)).reshape((-1,))[0]

        if inner_product_i > 0:
            gamma_i = 1
        else:
            gamma_i = 0
        
        if self.T_i <= self.T_i_high:
            beta_i = 1
        else:
            beta_i = 0

        if self.T_i >= self.T_i_low + self.delta_i:
            alpha_i = 1
        elif self.T_i <= self.T_i_low + self.delta_i and self.T_i >= self.T_i_low:
            alpha_i = 0.5 * (1 - np.cos(np.pi * (self.T_i - self.T_i_low) / self.delta_i))
        else:
            alpha_i = 0

        Vd_star_mod = (gamma_i + alpha_i * (1 - gamma_i)) * Vd_star
        ev_mod = Vb - Vd_star_mod

        # Kd = np.sqrt(np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])) * 10
        Kd = self.Kd

        dx_ti = (beta_i / self.x_ti) * (gamma_i * inner_product_i + (ev_mod.T @ Kd @ ev_mod)[0,0]) \
                + (alpha_i / self.x_ti) * (1 - gamma_i) * inner_product_i
        
        self.x_ti = self.x_ti + dx_ti * self.dt 

        # GUFIC control law       

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        M_d = np.eye(6) * 10

        tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod)

        # print('FT Sensor Value:', Fe.reshape((-1,)))

        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1,1))

        ####### Save all the dummy variables
        self.Ff_list.append(F_f)
        self.Vb_list.append(Vb)
        self.Ff_activation.append(gamma_f + alpha_f*(1 - gamma_f))
        self.rho_list.append(rho)

        return tau_cmd.reshape((-1,))
    
    def set_hole_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'hole', pos, R)


if __name__ == "__main__":
    robot_name = 'indy7' 
    show_viewer = True
    angle = 0
    angle_rad = angle / 180 * np.pi
    tracking = 'line'  # None, 'circle', 'line'

    assert tracking in [None, 'circle', 'line']

    RE = RobotEnv(robot_name, show_viewer = show_viewer, max_time = 8, obs_type = 'pos', window_size = 1, hole_ori = 'default', 
                  use_ext_force = False, testing = None, act_type = 'minimal', reward_version = 'force_penalty',
                  hole_angle = angle_rad, fix_camera = False, tracking = tracking)
    p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list = RE.run()

    Ff_list = RE.Ff_list
    Vb_list = RE.Vb_list
    Ff_activation = RE.Ff_activation
    rho_list = RE.rho_list

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

    # Perform the inner_product_f value
    inner_product_f_arr = np.zeros((len(Vb_arr),1))
    for i in range(len(Vb_arr)):
        inner_product_f_arr[i] = (Vb_arr[i].T @ Ff_arr[i]).reshape((-1,))[0]

    # plot the force profile 
    plt.figure(1)
    plt.plot(Fe_arr[:,2])
    plt.plot(-Fd_arr[:,2])
    plt.ylabel('Force z direction')
    plt.xlabel('Time Step')

    plt.figure(2)
    plt.subplot(311)
    plt.plot(p_arr[:,0])
    plt.ylabel('x (m)')
    plt.subplot(312)
    plt.plot(p_arr[:,1])
    plt.ylabel('y (m)')
    plt.subplot(313)
    plt.plot(p_arr[:,2])
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


