import os

# import mujoco_py
import numpy as np
# from mujoco_py import functions

import mujoco

# object indicator in mujoco
MJ_SITE_OBJ = 6  # `site` objec
MJ_BODY_OBJ = 1  # `body` object
MJ_GEOM_OBJ = 5  # `geom` object
# geom types
MJ_CYLINDER = 5
MJ_BOX = 6
MJ_MESH = 7


def attach_viewer(sim):
    return mujoco.MjViewer(sim)


def set_state(model, data, qpos, qvel):
    assert qpos.shape == (model.nq, ) and qvel.shape == (model.nv, )
    # old_state = data.get_state()
    # new_state = mujoco.MjSimState(old_state.time, qpos, qvel, old_state.act,
    #                                  old_state.udd_state)
    # data.set_state(new_state)
    # model.forward()
    data.qpos[:] = qpos
    data.qvel[:] = qvel


def get_contact_force(mj_model, mj_data, body_name):
    bodyId = mujoco.mj_name2id(mj_model, MJ_BODY_OBJ, body_name)
    force_com = mj_data.cfrc_ext[bodyId, :]
    trn_force = force_com.copy()

    return np.hstack((trn_force[3:], trn_force[:3]))


def get_geom_pose(model, geom_name):
    """Return the geom pose (relative to parent body).

    :param mujoco_py.MjModel model:
    :param str geom_name:
    :return: position, quaternion
    :rtype: tuple(np.array(3), np.array(4))
    """
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    pos = model.geom_pos[geom_id, :]
    quat = model.geom_quat[geom_id, :]
    return pos, quat


def get_geom_size(model, geom_name):
    """Return the geom size.

    :param mujoco_py.MjModel model:
    :param str geom_name:
    :return: (radius, half-length, _) for cylinder geom, and
             (X half-size; Y half-size; Z half-size) for box geom
    :rtype: np.array(3)
    """
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    if model.geom_type[geom_id] == MJ_BOX or model.geom_type[
            geom_id] == MJ_CYLINDER:
        return model.geom_size[geom_id, :]
    else:
        return None


def get_geom_friction(model, geom_name):
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    return model.geom_friction[geom_id, :]

def get_body_mass(model, body_name):
    body_id = mujoco.mj_name2id(model, MJ_BODY_OBJ, body_name)
    return model.body_mass[body_id]


def get_body_pose(model, body_name):
    body_id = mujoco.mj_name2id(model, MJ_BODY_OBJ, body_name)
    return model.body_pos[body_id], model.body_quat[body_id]


def get_mesh_vertex_pos(model, geom_name):
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    assert model.geom_type[geom_id] == MJ_MESH
    mesh_id = model.geom_dataid[geom_id]
    first_vertex_id = model.mesh_vertadr[mesh_id]
    no_vertex = model.mesh_vertnum[mesh_id]
    vertex_pos = model.mesh_vert[first_vertex_id:first_vertex_id + no_vertex]
    return vertex_pos


def set_geom_size(model, geom_name, size):
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    model.geom_size[geom_id, :] = size


def set_body_mass(model, body_name, mass):
    body_id = mujoco.mj_name2id(model, MJ_BODY_OBJ, body_name)
    model.body_mass[body_id] = mass


def set_geom_friction(model, geom_name, friction):
    geom_id = mujoco.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    model.geom_friction[geom_id, :] = friction


def set_body_pose(model, body_name, pos, quat):
    body_id = mujoco.mj_name2id(model, MJ_BODY_OBJ, body_name)
    model.body_pos[body_id, :] = pos
    model.body_quat[body_id, :] = quat

def set_body_pose_rotm(model, body_name, pos, R):
    quat = mat2quat(R)
    body_id = mujoco.mj_name2id(model, MJ_BODY_OBJ, body_name)
    model.body_pos[body_id, :] = pos
    model.body_quat[body_id, :] = quat


# -------- GEOMETRY TOOLs
def quat_error(q1, q2):
    """Compute the rotation vector (expressed in the base frame), that if follow
        in a unit time, will transform a body with orientation `q1` to
        orientation `q2`

    :param list/np.ndarray q1: Description of parameter `q1`.
    :param list/np.ndarray q2: Description of parameter `q2`.
    :return: a 3D rotation vector
    :rtype: np.ndarray

    """
    if isinstance(q1, list):
        q1 = np.array(q1)

    if isinstance(q2, list):
        q2 = np.array(q2)

    dtype = q1.dtype
    neg_q1 = np.zeros(4, dtype=dtype)
    err_rot_quat = np.zeros(4, dtype=dtype)
    err_rot = np.zeros(3, dtype=dtype)

    if q1.dot(q2) < 0:
        q1 = -q1

    mujoco.mju_negQuat(neg_q1, q1)
    mujoco.mju_mulQuat(err_rot_quat, q2, neg_q1)
    mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
    return err_rot


def quat2mat(q):
    """Tranform a quaternion to rotation amtrix.

    :param type q: Description of parameter `q`.
    :return: 3x3 rotation matrix
    :rtype: np.array
    """
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape((3, 3))


def pose_transform(p1, q1, p21, q21):
    """Coordinate transformation between 2 frames

    :param np.ndarray p1: position in frame 1
    :param np.ndarray q1: orientation (quaternion) in frame 1
    :param np.ndarray p21: relative position between frame 1 and 2
    :param np.ndarray q21: relative orientation between frame 1 and 2
    :return: position and orientation in frame 2
    :rtype: type

    """
    # quat to rotation matrix
    R21 = quat2mat(q21)

    p2 = p21 + R21.dot(p1)
    q2 = np.zeros_like(q1)
    mujoco.mju_mulQuat(q2, q21, q1)  # q2 = q21*q1
    return p2, q2


def integrate_quat(q, r, dt):
    """Integrate quaternion by a fixed angular velocity over the duration dt.

    :param np.array(4) q: quaternion.
    :param np.array(3) r: angular velocity.
    :param float dt: duration.
    :return: result quaternion.
    :rtype: np.array(4)
    """
    qres = np.zeros(4)
    qe = np.zeros(4)
    r = r * dt
    angle = np.linalg.norm(r)
    if angle < 1e-9:
        # if angle too small then return current q
        return q.copy()
    axis = r / angle
    mujoco.mju_axisAngle2Quat(qe, axis, angle)
    mujoco.mju_mulQuat(qres, qe, q)
    return qres


def transform_spatial(v1, q21):
    """Coordinate transformation of a spatial vector. The spatial vector can be either
    twist (linear + angular velocity) or wrench (force + torque)

    :param type v1: Spatial vector in frame 1
    :param type q21: transformation matrix (in terms of quaternion)
    :return: Description of returned object.
    :rtype: type
    """
    R21 = quat2mat(q21)
    R = np.block([[R21, np.zeros((3, 3))], [np.zeros((3, 3)), R21]])
    return R.dot(v1)


def similarity_transform(A1, q21):
    """Similarity transformation of a matrix from frame 1 to frame 2
            A2 = R21 * A1 * R12

    :param np.array((3, 3)) A1: 3x3 matrix.
    :param np.array(4) q21: quaternion representation.
    :return: 3x3 matrix
    :rtype: np.array

    """
    R21 = quat2mat(q21)
    return R21.dot(A1.dot(R21.T))


# NOTE: there are infinite rotation vector solutions for a particular
# orientation, the `ref` is to find the closest solution to a reference.
# Is there another minimal representation that could avoid this?
def quat2vec(q, ref=None):
    """Transform quaternion representation to rotation vector representation"""
    r = np.zeros(3)
    scale = 1
    mujoco.mju_quat2Vel(r, q, scale)
    if ref is not None:
        if r.dot(ref) < 0:
            angle = np.linalg.norm(r)
            r = r / angle
            angle = angle - 2 * np.pi
            r = r * angle
    return r


def inverse_frame(p, q):
    pi, qi = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(pi, qi, p, q)
    return pi, qi


def mat2quat(R):
    R = R.flatten()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R)
    return q


def mul_quat(q1, q2):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q1, q2)
    return q
