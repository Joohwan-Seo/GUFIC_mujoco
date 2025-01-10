import mujoco
import numpy as np

import time

class GaussNewtonIK:
    
    def __init__(self, model, data, step_size, tol, alpha, body_id, viewer = None):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        self.jacp = jacp
        self.jacr = jacr

        self.step_cnt = 0
        self.body_id = body_id
        self.max_cnt = 1000

        self.viewer = viewer
        self.IK_viewing = False
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
            
    def reset(self):
        self.step_cnt = 0
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
    
    # Gauss-Newton pseudocode implementation
    def calculate(self, pd, Rd, init_q):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_step(self.model, self.data)
        p = self.data.body(self.body_id).xpos
        R = self.data.body(self.body_id).xmat.reshape((3,3))
        ep = (p - pd).reshape((-1,1))
        Rd1 = Rd[:,0]; Rd2 = Rd[:,1]; Rd3 = Rd[:,2]
        R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

        eR = -0.5*((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

        error = np.vstack((ep, eR))

        if self.viewer is not None and self.IK_viewing:
            self.viewer.sync()
            time.sleep(0.05)

        while (np.linalg.norm(error) >= self.tol) and self.step_cnt < self.max_cnt:
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, p, self.body_id)
            jac = np.vstack((self.jacp, self.jacr))
            jac = jac[:,:6]
            #calculate delta of joint q
            product = jac.T @ jac
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jac.T
            else:
                j_inv = np.linalg.inv(product) @ jac.T
            
            delta_q = -jac.T @ error
            #compute next step
            self.data.qpos[:6] += self.step_size * delta_q.reshape((-1,))
            #check limits
            self.check_joint_limits(self.data.qpos[:6])
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            p = self.data.body(self.body_id).xpos
            R = self.data.body(self.body_id).xmat.reshape((3,3))
            ep = (p - pd).reshape((-1,1))
            Rd1 = Rd[:,0]; Rd2 = Rd[:,1]; Rd3 = Rd[:,2]
            R1 = R[:,0]; R2 = R[:,1]; R3 = R[:,2]

            eR = -0.5*((np.cross(R1,Rd1) + np.cross(R2,Rd2) + np.cross(R3,Rd3))).reshape((-1,1))

            error = np.vstack((ep, eR)) 
            self.step_cnt += 1

            if self.viewer is not None and self.IK_viewing:
                self.viewer.sync()
                time.sleep(0.05)

            
        self.data.qpos[-1] = 0
        self.data.qpos[-2] = 0
        mujoco.mj_forward(self.model, self.data) 
        print(f"Steps: {self.step_cnt}, error: {np.linalg.norm(error)}")
        print("q pos", self.data.qpos)