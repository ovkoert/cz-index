import numpy as np
import scipy
import copy


"""
Variables and functions for symplectic forms and pairing
"""
def get_omega0(dim):
    J0 = np.zeros((dim, dim))
    J0[dim//2:dim,0:dim//2] = np.identity(dim//2)
    J0[0:dim//2,dim//2:dim] = -np.identity(dim//2)
    return J0

eps = 1e-5
Omega6 = get_omega0(6)
Omega4 = get_omega0(4)

def dot(v, w):
    return np.sum(v * w)

def sdot4(v, w):
    return np.sum(v * (Omega4 @ w) )

def sdot6(v, w):
    return np.sum(v * (Omega6 @ w) )



def retract_sympl_path(path) -> np.array:
    r"""Retracts a symplectic path onto U(n), and takes its determinant

    Parameters
    ----------
    path : numpy.array
    Array with shape (N,2n,2n)

    Returns
    -------
    numpy.array
    Array containing the path of complex determinants
    """
    N = len(path)
    dim = path.shape[1]
    e_vals = np.zeros(path.shape)
    for i in range(N):
        Ah = np.asarray(path[i], dtype=float) #np.float128)
        V, Sigma, W = scipy.linalg.svd(Ah)
        e_vals[i] = np.asarray(V @ W, dtype=float)
    cdets = np.zeros(N, dtype=complex)
    for i in range(N):
        det = (np.linalg.det(e_vals[i][0:dim//2,0:dim//2]+1j * e_vals[i][dim//2:dim,0:dim//2] ))
        cdets[i] = det
    return cdets



def iwasawa(S):
    r"""Returns Iwasawa decomposition of a symplectic matrix, following
    Benzi, M., Razouk, N.: On the Iwasawa decomposition of a symplectic matrix. 
    Appl. Math. Lett. 20 (3), 260–265 (2007)

    Parameters
    ----------
    path : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    numpy.array
    array with shape (2n,2n) containing K-part
    numpy.array
    array with shape (2n,2n) containing A-part
    numpy.array
    array with shape (2n,2n) containing N-part
    """
    dim = len(S)
    n = dim // 2
    S1 = S[:, 0:n]
    Q, R = np.linalg.qr(S1)
    H = np.zeros((n,n))
    for i in range(n):
        H[i, i] = 1 / R[i, i]
    U = H @ R
    D = np.zeros((n,n))
    sD = np.zeros((n,n))
    sD_inv = np.zeros((n,n))
    for i in range(n):
        H[i, i] = R[i, i]
        D[i, i] = R[i, i] * R[i, i]
        sD[i,i] = np.sqrt(D[i, i])
        sD_inv[i,i] = 1 / sD[i,i]
    A = np.zeros((dim, dim))
    A[0:n,0:n] = sD
    A[n:dim,n:dim] = sD_inv
    K = np.zeros((dim, dim))
    K[0:n, 0:n] = Q[0:n,0:n] @ H @ sD_inv
    K[0:n, n:dim] = -Q[n:dim,0:n] @ H @ sD_inv
    K[n:dim, n:dim] = K[0:n, 0:n]
    K[n:dim, 0:n] = -K[0:n, n:dim]
    N = np.zeros((dim, dim))
    N[0:n, 0:n] = U
    N[:, n:dim] = np.linalg.inv(A) @ (K.transpose() @ S[:,n:dim])
    return K, A, N



def KtoU(K):
    r"""Converts 2n by 2n U(n) matrix to n by n complex form

    Parameters
    ----------
    path : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    numpy.array
    array with shape (n,n) containing U(n) matrix
    """
    n = len(K) // 2
    #U = np.zeros((n,n), dtype=complex)
    U = K[0:n,0:n] -1j * K[n:2*n,0:n]
    return U



def UtoK(U):
    """Converts n by n U(n) matrix to 2n by 2n real form

    Parameters
    ----------
    path : numpy.array
    Array with shape (n,n)

    Returns
    -------
    numpy.array
    array with shape (2n,2n) containing U(n) matrix
    in real form
    """
    n = len(U)
    K = np.zeros((2*n,2*n))
    K[0:n,0:n] = np.real(U)
    K[n:2*n,n:2*n] = np.real(U)
    K[n:2*n,0:n] = -np.imag(U)
    K[0:n,n:2*n] = +np.imag(U)
    return K

def maslov_func(S):
    """Returns the value of the Maslov function det(id-S)

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    float
    """
    return np.linalg.det(np.identity(4)-S)




"""Functions for dealing with elliptic matrices
These are symplectic matrices with eigenvalues on the unit circle.

"""


""" n by n U(n) matrix to 2n by 2n real form

Parameters
----------
path : numpy.array
    Array with shape (n,n)

Returns
-------
numpy.array
    array with shape (2n,2n) containing U(n) matrix
    in real form
"""




def find_elliptic_pairs(S):
    """Returns the elliptic pairs of eigenvalues of a symplectic matrix S.
    Since elliptic pairs are supposed to be on the unit circle and not degenerate,
    we discard values that are further away than eps

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    Tuple consisting of a list with the indices of the elliptic pairs,
    a numpy array with the eigenvalues, and
    a numpy array with the basis change (from diagonalization)
    """
    eigvals, B = np.linalg.eig(S)
    c_pairs = []
    indices = []
    for i in range(len(eigvals)):
        eigval = eigvals[i]
        if np.abs(eigval-1) < eps or np.abs(eigval+1)< eps or np.abs(np.abs(eigval)-1) > eps or i in indices:
            continue
        for j in range(i + 1,len(eigvals)):
            if np.abs( np.conjugate(eigvals[j]) - eigval ) < eps:
                indices.append(i)
                indices.append(j)
                c_pairs.append([i,j])
                break
    return c_pairs, eigvals, B



def path_removing_elliptic_pairs(S, steps=200):
    """Returns path of symplectic matrices starting at S and ending at S', where
    S' has no elliptic pairs left

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)
    steps : int

    Returns
    -------
    path : numpy.array with shape (steps, 2n, 2n)
    """
    c_pairs, eigvals, B = find_elliptic_pairs(S)
    if len(c_pairs) == 0:
        return np.array( [S for i in range(steps)])
    Binv = np.linalg.inv(B)
    angle_paths = []
    for pair in c_pairs:
        eigval = eigvals[pair[0]]
        angle = np.arctan2(np.imag(eigval), np.real(eigval))
        angle_paths.append(np.exp(np.linspace(angle * 1j, np.pi * 1j, steps) ) )
    angle_paths = np.asarray(angle_paths)
    lst_all = np.tile(eigvals, (steps, 1)).astype(complex)
    for idx, pair in enumerate(c_pairs):
        lst_all[:, pair[0]] = angle_paths[idx]
        lst_all[:, pair[1]] = np.conjugate(angle_paths[idx])
    return np.real((B * lst_all[:, np.newaxis, :]) @ Binv)




"""Functions for dealing with hyperbolic matrices
These are symplectic matrices with eigenvalues in norm greater or less than 1,
"""

def find_real_hyperbolic_pairs(S):
    """Returns the pairs of real eigenvalues of a symplectic matrix S.
    Eigenvalues with imaginary part larger than eps, or eigenvalues that are closer
    than eps to 1 are discarded.

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    Tuple consisting of a list with the indices of real hyperbolic pairs,
    a numpy array with the eigenvalues, and
    a numpy array with the basis change (from diagonalization)
    """
    eigvals, B = np.linalg.eig(S)
    eigvals = [np.real(val) if np.abs(np.imag(val) ) < eps else val for val in eigvals]
    c_pairs = []
    indices = []
    for i in range(len(eigvals)):
        eigval = eigvals[i]
        if np.abs(np.real(eigval) - eigval) > eps or np.abs(eigval+1)< eps or np.abs(eigval-1)< eps or i in indices:
            continue
        for j in range(i + 1,len(eigvals)):
            if np.abs( 1 / eigvals[j] - eigval ) < eps:
                indices.append(i)
                indices.append(j)
                c_pairs.append([i, j])
                break
    return c_pairs, eigvals, B


def path_removing_negative_hyperbolic_pairs(S, steps=200):
    r"""Returns path of symplectic matrices starting at S and ending at S', where
    S' has no negative real hyperbolic pairs left

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)
    steps : int

    Returns
    -------
    path : numpy.array with shape (steps, 2n, 2n)
    """
    c_pairs, eigvals, B = find_real_hyperbolic_pairs(S)
    eigvals = [np.real(val) if np.abs(np.imag(val) ) < eps else val for val in eigvals]
    if len(c_pairs) == 0:
        return np.array( [S for i in range(steps)])

    Binv = np.linalg.inv(B)
    eigenval_paths = []
    for pair in c_pairs:
        eigval = eigvals[pair[0]]
        if eigval < 0:
            eigenval_paths.append(np.linspace(eigval, -1.0, steps))
        else:
            eigenval_paths.append(np.linspace(eigval, eigval, steps))
    eigenval_paths = np.asarray(eigenval_paths)
    lst_all = np.array(np.tile(eigvals, (steps, 1)), dtype=complex)
    for idx, pair in enumerate(c_pairs):
        lst_all[:, pair[0]] = eigenval_paths[idx]
        lst_all[:, pair[1]] = 1.0 / eigenval_paths[idx]
    return (B * lst_all[:, np.newaxis, :]) @ Binv




def find_complex_hyperbolic_tuple(S):
    r"""Returns a tuple of complex eigenvalues of a symplectic matrix S
    that do not lie on the unit circle. Eigenvalues that are closer
    than eps to the unit circle are discarded.

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)

    Returns
    -------
    Tuple consisting of a list with the indices of complex hyperbolic eigenvalues,
    a numpy array with the eigenvalues, and
    a numpy array with the basis change (from diagonalization)
    """
    eigvals, B = np.linalg.eig(S)
    c_pairs = []
    for i in range(len(eigvals)):
        eigval = eigvals[i]
        if np.abs(np.real(eigval) - eigval) < eps or np.abs(np.abs(eigval)-1) < eps:
            continue
        c_pairs.append(i)
        #inverse
        for j in range(i + 1,len(eigvals)):
            if np.abs( 1 / eigvals[j] - eigval ) < eps:
                c_pairs.append(j)
                break
        #conjugate
        for j in range(i + 1,len(eigvals)):
            if np.abs( np.conjugate(eigvals[j]) - eigval ) < eps:
                c_pairs.append(j)
                break
        #conjugate inverse
        for j in range(i + 1,len(eigvals)):
            if np.abs( 1 / np.conjugate(eigvals[j]) - eigval ) < eps:
                c_pairs.append(j)
                break
        #pair_found = True
        break
    return c_pairs, eigvals,B



def path_removing_cpx_hyp_tuple(S, steps=200):
    r"""Returns path of symplectic matrices starting at S and ending at S',
    where S' has no complex hyperbolic eigenvalues left

    Parameters
    ----------
    S : numpy.array
    Array with shape (2n,2n)
    steps : int

    Returns
    -------
    path : numpy.array with shape (steps, 2n, 2n)
    """
    tuple, eigvals, B = find_complex_hyperbolic_tuple(S)
    if len(tuple) < 4:
        return np.array( [S for i in range(steps)])
    eigval = eigvals[tuple[0]]
    norm = np.linalg.norm(eigval)
    angle = np.arctan2(np.imag(eigval), np.real(eigval))
    Binv = np.linalg.inv(B)
    angle_path = np.exp(np.linspace(angle * 1j, np.pi * 1j, steps) )
    norm_path = np.linspace(norm, 1.0, steps)
    combined = norm_path * angle_path
    lst_all = np.tile(eigvals, (steps, 1)).astype(complex)
    lst_all[:, tuple[0]] = combined
    lst_all[:, tuple[1]] = 1.0 / combined
    lst_all[:, tuple[2]] = norm_path * np.conjugate(angle_path)
    lst_all[:, tuple[3]] = 1.0 / (norm_path * np.conjugate(angle_path))
    return np.real((B * lst_all[:, np.newaxis, :]) @ Binv)




def rotate4(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, -s, 0, 0],
                  [s, c, 0, 0],
                  [0, 0, c, -s],
                  [0, 0, s, c]])
    return R

def change_order_mat(B, target_order):
    P = np.zeros((4, 4))
    for i in range(4):
        P[i, target_order[i]] = 1.0
    return P

# fixed: use tolerance-based comparison with break after first match
# to correctly handle near-duplicate eigenvalues.
def ordering_values(vals):
    sorted_vals = sorted(copy.deepcopy(vals), reverse=True)
    ordering = []
    used = [False] * len(vals)
    for val in vals:
        for i in range(len(vals)):
            if not used[i] and np.abs(sorted_vals[i] - val) < eps:
                ordering.append(i)
                used[i] = True
                break
    return ordering, sorted_vals

# fixed from v1
# In v1, this function still worked correctly in most cases due to the ordering that 
# invoked eigenvalue computation algorithm used 
def path_removing_positive_hyperbolic_tuple(S, steps=200):
    c_pairs, eigvals, B = find_real_hyperbolic_pairs(S)
    eigvals = [np.real(val) if np.abs(np.imag(val) ) < eps else val for val in eigvals]
    if len(c_pairs) < 2 or np.min(np.real(eigvals) ) < 0:
        return path_removing_cpx_hyp_tuple(S, steps=steps)

    # fixed: save original eigenvalues before ordering_values overwrites them.
    original_eigvals = list(eigvals)
    ordering, eigvals = ordering_values(eigvals)

    # fixed: use the inverse permutation so that col j of BO gets the
    # eigenvector whose sorted rank is j (not the eigenvector at original index j).
    # inv_ordering[rank] = original index of the eigenvalue with that rank.
    inv_ordering = [0] * len(ordering)
    for k, rank in enumerate(ordering):
        inv_ordering[rank] = k

    O = np.zeros((4, 4))
    O[inv_ordering[0], 0] = 1   # col 0 ← eigvec for largest  (λ1)
    O[inv_ordering[1], 1] = 1   # col 1 ← eigvec for 2nd largest (λ2)
    O[inv_ordering[3], 2] = 1   # col 2 ← eigvec for smallest (1/λ1, symplectic pair of col 0)
    O[inv_ordering[2], 3] = 1   # col 3 ← eigvec for 2nd smallest (1/λ2, symplectic pair of col 1)

    BO = B @ O
    BO[:,0] = -1 / sdot4(BO[:,0], BO[:,2]) * BO[:,0]
    BO[:,1] = -1 / sdot4(BO[:,1], BO[:,3]) * BO[:,1]
    BOi = np.linalg.inv(BO)

    # fixed: use original_eigvals so that ODO reflects the true eigenvalue
    # diagonal in the BO basis, ensuring path1[0] = S.
    ODO = np.linalg.inv(O) @ np.diag(original_eigvals) @ O

    ts = np.linspace(0.0, 1.0, steps)
    lambda_s = np.linspace(ODO[0,0], 2.0, steps)
    mu_s = np.linspace(ODO[1,1], 2.0, steps)
    diag_vals1 = np.stack([lambda_s, mu_s, 1.0/lambda_s, 1.0/mu_s], axis=1)
    path1 = np.real((BO * diag_vals1[:, np.newaxis, :]) @ BOi)
    D44 = np.diag([2.0, 2.0, 0.5, 0.5])
    A_rot = np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]], dtype=float)
    M1 = BO @ D44 @ BOi
    M2 = BO @ A_rot @ D44 @ BOi
    phi_vals = 0.1 * ts
    path2 = np.real(np.cos(phi_vals)[:, np.newaxis, np.newaxis] * M1
                  + np.sin(phi_vals)[:, np.newaxis, np.newaxis] * M2)
    path3 = path_removing_cpx_hyp_tuple(path2[-1], steps=steps)
    return np.concatenate((path1, path2, path3))


def reduce_symplectic_matrix(S, steps=200):
    path2H = path_removing_positive_hyperbolic_tuple(S, steps=steps)
    pathE = path_removing_elliptic_pairs(path2H[-1], steps=steps)
    pathH = path_removing_negative_hyperbolic_pairs(pathE[-1], steps=steps)
    path = np.concatenate((path2H, pathE, pathH))
    return path




def interpolate_K(K, dim, steps):
    U = KtoU(K)
    eigU, basis = np.linalg.eig(U)
    basis_inv = np.linalg.inv(basis)
    angles = np.log(eigU)
    target_angles = np.zeros_like(angles)
    diags = np.linspace(angles, target_angles, steps)   # (steps, n)
    exp_diags = np.exp(diags)                           # (steps, n)
    tmp = (basis * exp_diags[:, np.newaxis, :]) @ basis_inv   # (steps, n, n)
    n = dim // 2
    Ks = np.zeros((steps, dim, dim))
    Ks[:, 0:n, 0:n] = np.real(tmp)
    Ks[:, n:dim, n:dim] = np.real(tmp)
    Ks[:, n:dim, 0:n] = -np.imag(tmp)
    Ks[:, 0:n, n:dim] = np.imag(tmp)
    return Ks


def interpolate_A(A, dim, steps):
    n = dim // 2
    A11_lin = np.linspace(A[0:n, 0:n], np.identity(n), steps)   # (steps, n, n)
    As = np.zeros((steps, dim, dim))
    As[:, 0:n, 0:n] = A11_lin
    As[:, n:dim, n:dim] = np.linalg.inv(A11_lin)   # batched inverse
    return As

def interpolate_N(N, dim, steps):
    n = dim // 2
    Ns = np.zeros((steps, dim, dim))
    N12_lin = np.linspace(N[0:n, n:dim], np.zeros((n, n)), steps // 2)
    Ns[:steps//2, 0:n, 0:n] = N[0:n, 0:n]
    Ns[:steps//2, n:dim, n:dim] = N[n:dim, n:dim]
    Ns[:steps//2, 0:n, n:dim] = N12_lin
    mid = np.zeros((dim, dim))
    mid[0:n, 0:n] = N[0:n, 0:n]
    mid[n:dim, n:dim] = N[n:dim, n:dim]
    Ns[steps//2:] = np.linspace(mid, np.identity(dim), steps - steps//2)
    return Ns



def path_hyperbolic_basepoint(S, steps=200):
    c_pairs, eigvals, B = find_real_hyperbolic_pairs(S)
    if len(c_pairs) != 1:
        return np.asarray([S])
    pair = c_pairs[0]
    B1 = copy.deepcopy(B)
    B1[:,pair[0]] /= -np.dot(B[:,pair[0]], Omega4 @ B[:,pair[1]] )
    other_pair = [0, 1, 2, 3]
    for el in pair:
        other_pair.remove(el)
    B1[:,other_pair[0]] /= -np.dot(B[:,other_pair[0]], Omega4 @ B[:,other_pair[1]] )
    B1 = np.real(B1)
    idx_order = [0]
    first_pair = other_pair
    second_pair = pair
    if 0 in pair:
        first_pair = pair
        second_pair = other_pair
    idx_order.append(second_pair[0])
    idx_order.append(first_pair[1])
    idx_order.append(second_pair[1])
    B2 = np.zeros((4,4))
    R = np.zeros((4,4))
    for i in range(4):
        B2[:,i] = B1[:, idx_order[i]]
        R[i,idx_order[i]] = 1.0
    Rinv = np.linalg.inv(R)
    B1 = B1 @ R
    K, A, N = iwasawa(B1)
    Kpath = interpolate_K(K, len(K), steps)
    Apath = interpolate_A(A, len(A), steps)
    Npath = interpolate_N(N, len(N), steps)
    Bpath = Kpath @ Apath @ Npath   # batched matmul, shape (steps, 4, 4)
    Bi = Bpath @ Rinv               # (steps, 4, 4)
    Bi_inv = np.linalg.inv(Bi)     # batched inverse
    Spath = np.real(Bi @ np.diag(eigvals) @ Bi_inv)
    nf = Spath[-1] #normal form

    # fixed: restrict argmax and max to the diagonal so that numerical
    # off-diagonal noise cannot select a wrong index or starting eigenvalue.
    diag_nf = np.diag(nf)
    idx0 = int(np.argmax(diag_nf))
    idx1 = idx0 - 2
    if idx0 + 2 < 4:
        idx1 = idx0 + 2
    rot_steps = steps // 10
    eigenval_path = np.linspace(np.max(diag_nf), 2.0, rot_steps)
    Rpath = np.tile(nf, (rot_steps, 1, 1))
    Rpath[:, idx0, idx0] = eigenval_path
    Rpath[:, idx1, idx1] = 1.0 / eigenval_path
    return np.real(np.concatenate((Spath, Rpath)) )




def compute_angular_index(cdets, T):
    angle = 0
    for i in range(T):
        delta = cdets[i+1] / cdets[i]
        angle += np.imag(np.log(delta))
    return int(np.round(angle / np.pi) )



def check_symplecticity(s_path):
    vals = []
    for A in s_path:
        vals.append(np.sum(np.abs(A.transpose() @ Omega4 @ A -Omega4) ) )
    return vals

def check_maslov_intersections(s_path):
    maslov_ints = []
    for i in range(len(s_path) - 1):
        maslov_ints.append(maslov_func(s_path[i+1]) * maslov_func(s_path[i]) )
    return maslov_ints

def check_continuity(s_path):
    jumps = []
    for i in range(len(s_path) - 1):
        jumps.append(np.sum(np.abs(s_path[i+1] - s_path[i] ) ) )
    return jumps


def get_index_sympl_path(s_path, steps=20000, error_report=False):
    path_reduced = reduce_symplectic_matrix(s_path[-1], steps=steps)
    S_reduced = np.real(path_reduced[-1])
    second_path = path_hyperbolic_basepoint(S_reduced, steps=steps)
    concat_path = np.concatenate((path_reduced, second_path))
    extend_s_path = np.concatenate( (s_path, concat_path) )
    cdets = retract_sympl_path(extend_s_path)
    CZ = compute_angular_index(cdets, len(cdets)-1)
    # For reliability checking
    if error_report:
        symplecticity = np.max( check_symplecticity(extend_s_path))
        extension_sign = np.min(check_maslov_intersections(concat_path))
        continuity = np.max(check_continuity(extend_s_path))
        print("Error in symplecticity was at most", symplecticity)
        if extension_sign > 0:
            print("Extention stayed away from Maslov cycle; closest at value", extension_sign)
        else:
            print("Extention crossed Maslov cycle; refine number of steps, eps", extension_sign)
        print("Error in continuity was at most", continuity, "; increase number of steps if this is too large.")
    return CZ


"""Functions for dealing with paths in Sp(2)"""

def parity_type(s_path_xi, T):
    trace = np.matrix.trace(s_path_xi[T])
    if trace < 2:
        return 1
    else:
        return 0

def return_closest_even(v):
    n = int(np.floor(v) )
    if n%2 == 0:
        return n
    else:
        return n+1

def return_closest_odd(v):
    n = int(np.floor(v) )
    if n%2 == 1:
        return n
    else:
        return n+1

def return_parity_corrected(v, parity):
    n = int(np.floor(v) )
    if n%2 == parity:
        return n
    else:
        return n+1

def compute_fractional_angular_index(cdets, T):
    angle = 0
    for i in range(T):
        delta = cdets[i+1] / cdets[i]
        angle += np.imag(np.log(delta))
    return (angle / np.pi)

def planar_index(p_path, T):
    cdets = retract_sympl_path(p_path)
    angular_idx = compute_fractional_angular_index(cdets, T)
    planar_CZ = return_parity_corrected(angular_idx, parity_type(p_path, T))
    return planar_CZ
