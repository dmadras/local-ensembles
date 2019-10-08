import numpy as np
import scipy.linalg

def lanczos_iteration(A_fn, dim, num_iters, eps=1e-8, dtype=np.float32, two_reorth=False):
    Q = [np.zeros((dim, 1)).astype(dtype)] # will collect Lanczos vectors here
    Beta = [1] # collect off-diagonal terms
    Alpha = [np.array([[0.]])] # collect diagonal terms

    # initialize
    q = np.random.uniform(size=(dim, 1)).astype(dtype)
    q /= np.linalg.norm(q)
    r = q
    Q.append(q)
    Q_k_range = Q[0]
    for k in range(1, num_iters + 1):
        z = A_fn(Q[k])
        alpha = np.matmul(Q[k].T, z)
        Alpha.append(alpha)

        Q_k_range = np.concatenate([Q_k_range, Q[k]], axis=1)
        z_orth = np.sum(np.matmul(z.T, Q_k_range) * Q_k_range, axis=1, keepdims=True)
        z = z - z_orth

        if two_reorth:
            # re-orthogonalizing twice improves stability
            z_orth = np.sum(np.matmul(z.T, Q_k_range) * Q_k_range, axis=1, keepdims=True)
            z = z - z_orth
        r = z
        beta = np.linalg.norm(r)
        Beta.append(beta)
        q = r / Beta[k]
        Q.append(q)
        if (k - 1) % 100 == 0:
            print('{:d} Lanczos iterations complete.'.format(k))
        if beta < eps:
            break
        
    return Q, Beta, Alpha

def get_eigendecomposition_from_tridiagonal(alpha, beta):
    Alpha_array = np.array(alpha).squeeze()
    Beta_array = np.array(beta).squeeze()
    T_evals, T_evecs = scipy.linalg.eigh_tridiagonal(Alpha_array, Beta_array)
    return T_evals, T_evecs

def sort_eigendata_by_absolute_value(evals, evecs):
    einfo_by_abs = sorted(zip(evals, [evecs[:, i] for i in range(evecs.shape[1])]),
                       key=lambda x: abs(x[0]), reverse=True)
    evals_by_abs = [x[0] for x in einfo_by_abs]
    evecs_by_abs = np.stack([x[1] for x in einfo_by_abs], axis=1)
    return evals_by_abs, evecs_by_abs

