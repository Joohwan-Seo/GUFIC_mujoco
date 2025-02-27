import numpy as np
# from mujoco_py import functions

# This file was written in mujoco_py 2.0.2, I want to change it into mujoco 3.0

import mujoco
import os

from gufic_env.utils.filter import ButterLowPass
from gufic_env.utils.mujoco import (MJ_SITE_OBJ, get_contact_force,
                                    inverse_frame, pose_transform,
                                    transform_spatial)

# from ctypes import *

# import linear control package
import control as ctrl
from scipy.linalg import block_diag


class RobotState:
    """Wrapper to the mujoco sim to store robot state and perform
    simulation operations (step, forward dynamic, ...).

    :param mujoco.MjSim sim: mujoco sim
    :param str ee_site_name: name of the end-effector site in mujoco xml model.
    :attr mujoco.MjData data: Description of parameter `data`.
    :attr mujoco.MjModel model: Description of parameter `model`.
    """

    def __init__(self, model, data, ee_site_name, robot_name):
        self.data = data
        self.model = model
        self.robot_name = robot_name

        if self.robot_name == 'indy7':
            self.N = 6
        else:
            self.N = 7
        self.ee_site_idx = mujoco.mj_name2id(
            self.model, MJ_SITE_OBJ, ee_site_name)
        self.isUpdated = False
        # low pass filter
        dt = model.opt.timestep
        # print('dt:', dt)
        fs = 1 / dt
        # cutoff = 2 ## Default value is 50
        cutoff = 10
        self.fe = np.zeros(6)
        self.lp_filter = ButterLowPass(cutoff, fs, order=5)
        self.lp_filter_raw = ButterLowPass(cutoff=10, fs=fs, order=5)
        self.ft_body_name = "ft_assembly"
        # self.lp_filter2 = ButterLowPass(cutoff, fs, order=2)

        self.Ad, self.Bd = self.define_filter(5, dt)
        self.filter_state = np.zeros((12,1))

        self.Ad_raw, self.Bd_raw = self.define_filter(50, dt)
        self.filter_state_raw = np.zeros((12,1))

        print("initialization complete")

    def define_filter(self, cutoff, dt, dim =6):
        ws = cutoff
        A = np.array([[0, 1],
                      [-ws**2, -2 * 1 * ws]])
        B = np.array([[0], [ws**2]])
        C = np.array([[1, 0]])

        sys = ctrl.ss(A, B, C, 0)
        sys_d = ctrl.c2d(sys, dt)

        Ad1 = sys_d.A
        Bd1 = sys_d.B

        # stack Ad and Bd for dim times
        Ad = block_diag(*[Ad1 for _ in range(dim)])
        Bd = block_diag(*[Bd1 for _ in range(dim)])

        return Ad, Bd
    
    def lp_filter_implemented(self, force_torque):
        # 0, 2, 4, 6, 8, 10 indices are filtered values
        xf = self.filter_state[::2]

        # 1, 3, 5, 7, 9, 11 indices are filtered derivative values
        dxf = self.filter_state[1::2]

        self.filter_state = self.Ad @ self.filter_state + self.Bd @ force_torque.reshape((-1,1))

        return xf, dxf
    
    def lp_filter_implemented_raw(self, force_torque):
        # 0, 2, 4, 6, 8, 10 indices are filtered values
        xf = self.filter_state_raw[::2]

        # 1, 3, 5, 7, 9, 11 indices are filtered derivative values
        dxf = self.filter_state_raw[1::2]

        self.filter_state_raw = self.Ad_raw @ self.filter_state_raw + self.Bd_raw @ force_torque.reshape((-1,1))

        return xf, dxf


    def update(self):
        """Update the internal simulation state (kinematics, external force, ...).
        Should be called before perform any setters or getters"""
        # update position-dependent state (kinematics, jacobian, ...)
        mujoco.mj_step1(self.model, self.data)
        # udpate the external force internally
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self.isUpdated = True

    def update_dynamic(self):
        """Update dynamic state (forward dynamic). The control torque should be
        set between self.update() and self.update_dynamic()"""
        mujoco.mj_step2(self.model, self.data)
        self.isUpdated = False

    def is_update(self):
        return self.isUpdated

    def reset_filter_state(self):
        self.lp_filter.reset_state()

    def get_pose(self):
        p = self.data.site_xpos[self.ee_site_idx].copy()    # pos
        R = self.data.site_xmat[self.ee_site_idx].copy()    # rotation matrix

        return p, R.reshape((3,3))
    
    def get_ee_force(self,):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")

        # Get address and dimension of the sensor
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        force = np.copy(self.data.sensordata[adr:adr + dim])
        # get torque sensor data
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_sensor")
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        torque = np.copy(self.data.sensordata[adr:adr + dim])
        force_torque = np.concatenate([force, torque])
        # update robot state

        # if hasattr(self.lp_filter, 'zi'):
        #     ft = self.lp_filter.zi[0,:]
        #     dft = self.lp_filter.zi[1,:]
        # else:
        #     ft = np.zeros(6)
        #     dft = np.zeros(6)
        
        # self.force_sensor_data = self.lp_filter(force_torque.reshape((-1, 6)))[0, :]

        ft , dft = self.lp_filter_implemented(force_torque)
        return ft, dft

        # return ft, dft
    
    def get_ee_force_raw(self,):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")

        # Get address and dimension of the sensor
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        force = np.copy(self.data.sensordata[adr:adr + dim])
        # get torque sensor data
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_sensor")
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        torque = np.copy(self.data.sensordata[adr:adr + dim])
        force_torque = np.concatenate([force, torque])

        # force_sensor_data = self.lp_filter_raw(force_torque.reshape((-1, 6)))[0, :]

        ft_raw , dft_raw = self.lp_filter_implemented_raw(force_torque)

        return ft_raw , dft_raw
    
    def transform_rot(self, fe, desired):
        pe, Re = self.get_pose()
        ps, Rs = desired

        R12 = Rs.T @ Re
        Mat = np.block([[R12, np.zeros((3, 3))], [np.zeros((3, 3)), R12]])

        return Mat.dot(fe)
    
    def gauss_newton_IK(self, pd, Rd, init_q, step_size = 0.4, tol = 0.001, max_cnt = 500):
        self.data.qpos = init_q
        p, R = self.get_pose()
        ep = (p - pd).reshape((-1,1))
        Rd1 = Rd[:,0]; Rd2 = Rd[:,1]; Rd3 = Rd[:,2]
        R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

        eR = -0.5*((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

        error = np.vstack((ep, eR))

        step_cnt = 0

        while (np.linalg.norm(error) >= tol) and step_cnt < max_cnt:
            jac = self.get_jacobian()
            product = jac.T @ jac
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jac.T
            else:
                j_inv = np.linalg.inv(product) @ jac.T

            delta_q = -jac.T @ error
            #compute next step
            self.data.qpos[:6] += step_size * delta_q.reshape((-1,))
            #check limits
            self.check_joint_limits(self.data.qpos[:6])
            #compute forward kinematics
            self.update()
            #calculate new error
            p, R = self.get_pose()
            ep = (p - pd).reshape((-1,1))
            Rd1 = Rd[:,0]; Rd2 = Rd[:,1]; Rd3 = Rd[:,2]
            R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

            eR = -0.5*((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

            error = np.vstack((ep, eR)) 
            step_cnt += 1

        if self.model.nv == 8:
            self.data.qpos[-1] = 0
            self.data.qpos[-2] = 0
        elif self.model.nv == 6:
            pass
        mujoco.mj_forward(self.model, self.data) 
        print(f"Steps: {step_cnt}, error: {np.linalg.norm(error)}")
        print("q pos", self.data.qpos)

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))


    def get_jacobian(self):
        """Get 6x7 geometric jacobian matrix."""
        dtype = self.data.qpos.dtype
        N_full = self.model.nv
        jac = np.zeros((6, N_full), dtype=dtype)
        jac_pos = np.zeros((3 , N_full), dtype=dtype)
        jac_rot = np.zeros((3 , N_full), dtype=dtype)
        mujoco.mj_jacSite(
            self.model, self.data,
            jac_pos, jac_rot, self.ee_site_idx)
        jac[3:] = jac_rot.reshape((3, N_full))
        jac[:3] = jac_pos.reshape((3, N_full))
        # only return first 7 dofs
        return jac[:, :self.N].copy()

    def get_body_jacobian(self):
        Js = self.get_jacobian()

        p, R = self.get_pose()

        transform = np.block([[R.T, np.zeros((3,3))], [np.zeros((3,3)), R.T]])

        Jb = transform @ Js

        return Jb

    def get_body_ee_velocity(self):
        Jb = self.get_body_jacobian()
        dq = self.get_joint_velocity()[:self.N]
        Vb = Jb@dq.reshape((-1,1))

        return Vb
    
    def get_spatial_ee_velocity(self):
        Js = self.get_jacobian()
        dq = self.get_joint_velocity()

        Vs = Js@dq.reshape((-1,1))

        return Vs

    def get_joint_pose(self):
        return self.data.qpos.copy()

    def get_joint_velocity(self):
        return self.data.qvel.copy()

    def get_bias_torque(self):
        """Get the gravity and Coriolis, centrifugal torque """
        return self.data.qfrc_bias[:self.N].copy()
    
    def get_full_inertia(self):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        return M[:self.N, :self.N]

    def get_timestep(self):
        """Timestep of the simulator is timestep of controller."""
        return self.model.opt.timestep

    def get_sim_time(self):
        return self.data.time

    def set_control_torque(self, tau, gripper = 0):
        """Set control torque to robot actuators."""
        assert tau.shape[0] == self.N
        # self.data.ctrl[:] = np.hstack((tau, [0, 0]))
        # if self.robot_name == 'ur5e':
        #     self.data.ctrl[:] = tau
        # else:
        #     self.data.ctrl[:] = np.hstack((tau, [0, 0]))

        if self.robot_name == 'indy7':
            if self.model.nv == 8:
                self.data.ctrl[:] = np.hstack((tau, [-gripper, gripper]))
            elif self.model.nv == 6:
                self.data.ctrl[:] = tau
        else:
            self.data.ctrl[:] = tau
