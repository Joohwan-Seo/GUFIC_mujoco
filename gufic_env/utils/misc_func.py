import numpy as np

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





