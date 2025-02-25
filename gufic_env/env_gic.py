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



class RobotEnv:
    def __init__(self, robot_name = 'indy7', env_type = 'square_PIH', max_time = 15, show_viewer = False, 
                 obs_type = 'pos', hole_ori = 'default', testing = False, reward_version = None, window_size = 1, 
                 use_ext_force = False, act_type = 'default', mixed_obs = False, hole_angle = 0.0, in_dist = True, fix_camera = False):
        
        self.robot_name = robot_name
        print(self.robot_name)
        self.env_type = env_type
        self.hole_ori = hole_ori
        self.testing = testing 

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

        self.pd = np.array([0.50, 0.05, 0.15])
        self.Rd = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
        
        self.p_plate = np.array([0.50, 0, 0.1])
        self.R_plate = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])

        self.show_viewer = show_viewer
        self.load_xml()

        self.obs_type = obs_type
        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = 0.001
        self.max_iter = int(max_time/self.dt)

        self.time_step = 0

        self.Fe = np.zeros((6,1))
        self.reset()

    def load_xml(self):
        # dir = "/home/joohwan/deeprl/research/GIC_Learning_public/"
        dir = os.getcwd() + '/'
        if self.robot_name == 'ur5e':
            raise NotImplementedError

        elif self.robot_name == 'indy7':
            model_path = dir + "gufic_env/mujoco_models/Indy7_insertion.xml"

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

            p_init = self.pd.reshape((-1,1)) + self.Rd @ np.array([rand_xy[0], rand_xy[1], -0.2]).reshape(-1,1)
            R_init = self.Rd @ Rz @ Ry @ Rx

            p_init = p_init.reshape((-1,))
        else:
            rand_xy = 2*(np.random.rand(2,) - 0.5) * 0.04
            rand_rpy = 2*(np.random.rand(3,) - 0.5) * 12 /180 * np.pi

            Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
            Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
            Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

            p_init = self.pd.reshape((-1,1)) + self.Rd @ np.array([rand_xy[0], rand_xy[1], -0.2]).reshape(-1,1)
            R_init = self.Rd @ Rz @ Ry @ Rx

            p_init = p_init.reshape((-1,))

        # while True:
        #     action = np.array([1,1,1,1,1,1]) * 0.8
        #     obs, reward, done, info = self.step(action)

        #     eg = self.get_eg()
        #     ev = self.get_eV()

        #     count += 1

        #     if np.linalg.norm(eg) < 0.01 or count > 4000:
        #         if count > 3500:
        #             print('Emergency Break')
        #         break

        q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0, 0])

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
        
        q0 = np.array([0, 0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0, 0])

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
        return_arr = []        
        for i in range(self.max_iter):
            # print(i)
            # action = self.get_expert_action()
            # action = np.array([0,0,0,0,0,0])
            action = np.array([1000, 1000, 100, 750, 750, 750])
            obs, reward, done, info = self.step(action)

            return_arr.append(reward)
            # print(reward)

            if self.show_viewer:
                if i % 10 == 0:
                    self.viewer.sync()

            if done:
                break

            self.time_step = i

        print('total_Return:',sum(return_arr))

    def step(self, gains):
        self.robot_state.update()

        tau_cmd = self.geometric_impedance_control(gains)

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

    def geometric_impedance_control(self, gains):
        Jb = self.robot_state.get_body_jacobian()

        # M,C,G = self.robot_state.get_dynamic_matrices()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()

        #0 Get impedance gains
        Kp = np.diag(gains[0:3])
        KR = np.diag(gains[3:6])

        #1 Calculate positional force
        x, R = self.robot_state.get_pose()
        xd, Rd = self.pd, self.Rd

        fp = R.T @ Rd @ Kp @ Rd.T @ (x - xd).reshape((-1,1))
        # fp = Kp @ (x - xd).reshape((-1,1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)

        fg = np.vstack((fp,fR))

        #2. get error vel vector        
        eV = self.get_eV()
        Kd = np.sqrt(np.block([[Kp, np.zeros((3,3))],[np.zeros((3,3)), KR]])) * 10

        Fe_FT = self.get_FT_value().reshape((-1,1))

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)

        M_d = np.eye(6) * 10

        tau_tilde = M_tilde @ np.linalg.inv(M_d) @ (- Kd @ eV - fg + Fe_FT) - Fe_FT

        # print('FT Sensor Value:', Fe_FT.reshape((-1,)))

        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1,1))

        return tau_cmd.reshape((-1,))
    
    def set_hole_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'hole', pos, R)


if __name__ == "__main__":
    robot_name = 'indy7' 
    env_type = 'square_PIH'
    show_viewer = True
    angle = 0
    angle_rad = angle / 180 * np.pi
    RE = RobotEnv(robot_name, env_type, show_viewer = True, obs_type = 'pos', window_size = 1, hole_ori = 'default', 
                  use_ext_force = False, testing = True, act_type = 'minimal', reward_version = 'force_penalty',
                  hole_angle = angle_rad, fix_camera = False)
    RE.run()
