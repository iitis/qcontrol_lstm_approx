import numpy as np
import scipy
import tensorflow as tf
########################################################################################################################
########################################################################################################################
# Constants
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Pauli matrices
Sx = np.array([[0,1],
               [1,0]])

Sy = np.array([[0, -1j],
               [1j, 0]])

Sz = np.array([[1,0],
               [0,-1]])
########################################################################################################################
ketbra01 = np.array([[0, 1],
                    [0, 0]])
ketbra10 = np.array([[0, 0],
                    [1, 0]])
ketbra00 = np.array([[1, 0],
                      [0, 0]])
ketbra11 = np.array([[0, 0],
                      [0, 1]])
########################################################################################################################
# components of the Hamiltonians used for the control and for the noise
# Lc_x stands for sx on the first (left) qubit and id on the second
# Rc_y stands for id on the first qubit and sy on the second (right)
Lc_x = np.kron(Sx, np.eye(2))
Lc_y = np.kron(Sy, np.eye(2))
Lc_z = np.kron(Sz, np.eye(2))
Rc_y = np.kron(np.eye(2), Sy)

########################################################################################################################
# Function names follow the scheme of the system Hamiltonian
# For example, function
# sy_id_spinChain_2x1
# representes Hamiltonian
# H = \gamma sx\otimes id + H_0 + H_c
# where H_0 is the spin chain Hamiltonian
# and H_c is control Hamiltonian acting on the first qubit only
########################################################################################################################
def Sz_and_ketbra01_Lindbald_spinChain_drift(params):
    gamma, alpha = params
    L = [np.kron(Sz, np.eye(2)), np.kron(ketbra01, np.eye(2))]  # , np.kron(np.eye(2), ketbra01), np.kron(np.eye(2),Sz)]
    Lind_part = 0
    for i in range(len(L)):
        Lind_part += (
        2 * np.kron(L[i], L[i].conjugate()) - (np.kron(np.dot(L[i].conjugate().transpose() , L[i]), np.eye(4)) + np.kron(np.eye(4), L[i].transpose() * L[i].conjugate())))

    # Lind_part *= gamma

    spChain = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    Ham_part = np.kron(np.eye(4), spChain.conjugate()) - np.kron(spChain, np.eye(4))
    Ham_part *= -1j

    # drift plus noise
    drift = gamma * Lind_part + Ham_part


    # controls in hamiltonian form of lindblad equation
    Hc_x = np.kron(np.eye(4), Lc_x.conjugate()) - np.kron(Lc_x, np.eye(4))
    Hc_y = np.kron(np.eye(4), Lc_y.conjugate()) - np.kron(Lc_y, np.eye(4))

    ctrls = [-1j * Hc_x, -1j * Hc_y]
    return (ctrls, drift)


def id_aSxbSy_spinChain_2x1(params):
    gamma, alpha = params
    print("gamma= ", gamma)
    print("alpha= ", alpha)

    Hc_x = np.kron(np.eye(4), Lc_x.conjugate()) - np.kron(Lc_x, np.eye(4))
    Hc_z = np.kron(np.eye(4), Lc_z.conjugate()) - np.kron(Lc_z, np.eye(4))
    ctrls = [-1j * Hc_x, -1j * Hc_z]

    spChain = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    Ham_part = np.kron(np.eye(4), spChain.conjugate()) - np.kron(spChain, np.eye(4))
    Ham_part *= -1j

    aSxbSy = alpha * Sx + (1 - alpha) * Sy
    Rc_rnd = np.kron(np.eye(2), aSxbSy)
    H0 = np.kron(np.eye(4), Rc_rnd.conjugate()) - np.kron(Rc_rnd, np.eye(4))
    drift = gamma * (-1j * H0) + Ham_part

    return (ctrls, drift)

def aSxbSy_id_spinChain_dim_2x1(params):
    gamma, alpha = params
    # print("gamma= ", gamma)
    # print("alpha= ", alpha)

    Hc_x = np.kron(np.eye(4), Lc_x.conjugate()) - np.kron(Lc_x, np.eye(4))
    Hc_z = np.kron(np.eye(4), Lc_z.conjugate()) - np.kron(Lc_z, np.eye(4))
    ctrls = [-1j * Hc_x, -1j * Hc_z]

    spChain = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    Ham_part = np.kron(np.eye(4), spChain.conjugate()) - np.kron(spChain, np.eye(4))
    Ham_part *= -1j

    aSxbSy = alpha*Sx + (1-alpha)*Sy
    Lc_rnd = np.kron(aSxbSy, np.eye(2))
    H0 = np.kron(np.eye(4), Lc_rnd.conjugate()) - np.kron(Lc_rnd, np.eye(4))
    drift = gamma * (-1j * H0) + Ham_part

    return (ctrls, drift)

def spinChainDrift_spinChain_dim_2x1(params):
    gamma, alpha, beta = params
    print("gamma= ", gamma)
    print("alpha= ", alpha)
    print("beta= ", beta)

    Hc_x = np.kron(np.eye(4), Lc_x.conjugate()) - np.kron(Lc_x, np.eye(4))
    Hc_z = np.kron(np.eye(4), Lc_z.conjugate()) - np.kron(Lc_z, np.eye(4))
    ctrls = [-1j * Hc_x, -1j * Hc_z]

    spChain = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    Ham_part = np.kron(np.eye(4), spChain.conjugate()) - np.kron(spChain, np.eye(4))
    Ham_part *= -1j

    spinChainDrift = alpha*np.kron(Sx, Sx) + beta*np.kron(Sy, Sy) + (1-alpha-beta)*np.kron(Sz, Sz)

    H0 = np.kron(np.eye(4), spinChainDrift.conjugate()) - np.kron(spinChainDrift, np.eye(4))
    drift = gamma * (-1j * H0) + Ham_part

    return (ctrls, drift)

########################################################################################################################
########################################################################################################################
# Functions related to mathematical operations required during the learning
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Integration of the control pulses
# Returns: superoperator resulting from using the sequence of control pulses
# Flag tf_result is set to true if the integration should be executed using tf objects
########################################################################################################################
def integrate_lind(h, params, n_ts, evo_time, noise_name, tf_result):


    if noise_name == 'id_aSxbSy_spinChain_2x1':
        n = 16
        ctrls, drift = id_aSxbSy_spinChain_2x1(params)
    elif noise_name == "aSxbSy_id_spinChain_dim_2x1":
        n = 16
        ctrls, drift = aSxbSy_id_spinChain_dim_2x1(params)
    elif noise_name == "spinChainDrift_spinChain_dim_2x1":
        n = 16
        ctrls, drift = spinChainDrift_spinChain_dim_2x1(params)
    elif noise_name == "Sz_and_ketbra01_Lindbald_spinChain_drift":
        n = 16
        ctrls, drift = Sz_and_ketbra01_Lindbald_spinChain_drift(params)


    A = np.eye(n,dtype=complex)

    if tf_result:
        for i in range(n_ts):
            Hc = tf.convert_to_tensor(np.sum([h[i][j] * ctrls[j] for j in range(len(ctrls))], axis=0), dtype=tf.complex128)
            A = tf.matmul(matrixExp(evo_time / n_ts * (drift + Hc), 8), A, a_is_sparse=True,b_is_sparse=True)
            # A = tf.matmul(tf.linalg.expm(evo_time / n_ts * (drift + Hc)), A)
    else:
        for i in range(n_ts):
            Hc = np.sum([h[i][j] * ctrls[j] for j in range(len(ctrls))], axis=0)
            A = np.dot(scipy.linalg.expm(evo_time / n_ts * (drift + Hc)), A)

    return A

########################################################################################################################
#
########################################################################################################################
def matrixExp(X, precision):
    n = tf.shape(X)[1]
    powX = tf.reshape(tf.eye(n, dtype=tf.complex128), tf.shape(X))
    res = tf.reshape(tf.eye(n, dtype=tf.complex128), tf.shape(X))

    for i in range(1, precision):
        c = complex(i, 0)
        powX = tf.matmul(powX, X, a_is_sparse=True,b_is_sparse=True) / c
        res += powX
    return res

########################################################################################################################
#
########################################################################################################################
def fidelity_err(list_of_superops, dim, tf_result):
    target_superop = list_of_superops[0]
    generated_superop = list_of_superops[1]

    if tf_result:
        superop_diff = tf.subtract(target_superop, generated_superop)
        result = tf.real(tf.trace(tf.matmul(superop_diff, superop_diff, adjoint_a=True)) / (2 * dim ** 2))
        result = tf.cast(result, tf.float32)
        return result
    else:
        superop_diff = target_superop - generated_superop
        result = np.real(np.trace(np.dot(superop_diff.conjugate().transpose(), superop_diff)) / (2 * dim ** 2))
        return result
