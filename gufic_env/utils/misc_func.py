import numpy as np
import sympy as sp

def vee_map(R):
    v3 = -R[0,1]
    v1 = -R[1,2]
    v2 = R[0,2]
    return np.array([v1,v2,v3]).reshape((-1,1))

def hat_map(w):
    w = w.reshape((-1,))
    w_hat = np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
    return w_hat

def rotmat_x(th):
    R = np.array([[1,0,0],
                    [0,np.cos(th),-np.sin(th)],
                    [0,np.sin(th), np.cos(th)]])

    return R

def adjoint_g_ed(g_ed):
    p = g_ed[:3,3]
    R = g_ed[:3,:3]

    p_hat = hat_map(p)
    # translation part first adjoint map
    adj = np.zeros((6,6))
    adj[:3,:3] = R
    adj[3:,3:] = R
    adj[:3,3:] = p_hat @ R

    return adj

def adjoint_g_ed_dual(g_ed):
    mat = adjoint_g_ed(np.linalg.inv(g_ed))

    return mat.T

def adjoint_g_ed_deriv(g, gd, v, w, vd, wd):
    v = v.reshape((-1,1))
    w = w.reshape((-1,1))
    vd = vd.reshape((-1,1))
    wd = wd.reshape((-1,1))

    g_ed = np.linalg.inv(g) @ gd
    p_ed = g_ed[:3,3]
    R_ed = g_ed[:3,:3]

    mat = np.zeros((6,6))

    dR_ed = hat_map(w) @ R_ed - R_ed @ hat_map(wd)
    dp_ed = -v - hat_map(w) @ p_ed + R_ed @ vd

    mat[:3, :3] = dR_ed
    mat[:3, 3:] = hat_map(p_ed)@ dR_ed + hat_map(dp_ed) @ R_ed
    mat[3:, 3:] = dR_ed

    return mat

def initialize_trajectory(task, max_time = 10):
    t = sp.symbols('t')

    if task == "regulation":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "circle":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "line":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "sphere":
        pd_default = np.array([0.40, 0.0, 0.0]) #center of the sphere
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
    else:
        raise ValueError("Invalid task")

    pd_default_sym = sp.Matrix(pd_default)
    Rd_default_sym = sp.Matrix(Rd_default)

    if task == 'regulation':
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym
    elif task == 'circle':
        r = 0.1
        pd_t_sim = pd_default_sym + sp.Matrix([r * sp.cos(t), r * sp.sin(t), 0])
        Rd_t_sim = Rd_default_sym
        
    elif task == 'line':
        pd_t_sim = pd_default_sym + sp.Matrix([0, 0.05 * t, 0])
        Rd_t_sim = Rd_default_sym

    elif task == 'sphere':

        total_radian = 1/2*np.pi
        omega_value = total_radian / max_time

        theta_y = omega_value * t - total_radian * 0.5


        r_sphere = 0.304
        pd_t_sim = pd_default_sym + sp.Matrix([0, r_sphere * sp.sin(theta_y), -0.10 + r_sphere * sp.cos(theta_y)])
        rotmat_y = sp.Matrix([[sp.cos(-theta_y), 0, sp.sin(-theta_y)], [0, 1, 0], [-sp.sin(-theta_y), 0, sp.cos(-theta_y)]])
        Rd_t_sim = Rd_default_sym @ rotmat_y


    # Differentiate with symbolic expressions
    dpd_t_sim = sp.diff(pd_t_sim, t)
    dRd_t_sim = sp.diff(Rd_t_sim, t)
    ddpd_t_sim = sp.diff(dpd_t_sim, t)
    ddRd_t_sim = sp.diff(dRd_t_sim, t)

    # Convert symbolic to numpy expressions
    pd_t = sp.lambdify(t, pd_t_sim, "numpy")
    Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")
    dpd_t = sp.lambdify(t, dpd_t_sim, "numpy")
    dRd_t = sp.lambdify(t, dRd_t_sim, "numpy")
    ddpd_t = sp.lambdify(t, ddpd_t_sim, "numpy")
    ddRd_t = sp.lambdify(t, ddRd_t_sim, "numpy")

    return pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t

def set_gains(controller = "GUFIC", task = "regulation"):
    assert controller in ["GUFIC", "GIC"], "Invalid controller"
    assert task in ["regulation", "circle", "line", "sphere"], "Invalid task"
    
    if controller == "GIC":
        if task == "regulation":
            Kp = np.eye(3) * np.array([1500, 1500, 1500])
            KR = np.eye(3) * np.array([1500, 1500, 1500])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        elif task in ["circle", "line", "sphere"]:
            Kp = np.eye(3) * np.array([2500, 2500, 1500])
            KR = np.eye(3) * np.array([2000, 2000, 2000])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        return Kp, KR, Kd

    elif controller == "GUFIC":
        if task == "regulation":
            Kp = np.eye(3) * np.array([1500, 1500, 10])
            KR = np.eye(3) * np.array([1500, 1500, 1500])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            kp_force = 1.0
            kd_force = 0.5
            ki_force = 4.0

        elif task in ["circle", "line", "sphere"]:
            Kp = np.eye(3) * np.array([2000, 2000, 10])
            KR = np.eye(3) * np.array([2000, 2000, 2000])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            if task == "sphere":
                kp_force = 1.5
                kd_force = 0.75
                ki_force = 6.0

            else:
                kp_force = 1.0
                kd_force = 0.5
                ki_force = 4.0

        zeta = 5.0

        return Kp, KR, Kd, kp_force, kd_force, ki_force, zeta






