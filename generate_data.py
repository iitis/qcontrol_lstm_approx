import numpy as np
import pathlib
from multiprocessing import Pool
from functools import partial

# QuTiP imports
from qutip import Qobj, identity, tensor
# QuTiP control modules
import qutip.control.pulseoptim as cpo
# QuTiP logger
import qutip.logging_utils as logging
logger = logging.get_logger()
# set this to None or logging.WARN for 'quiet' execution
log_level = logging.ERROR

# local modules
from noise_models_and_integration import *
from constants_of_experiments import *




def rand_unitary(n):
    '''A Random unitary matrix distributed with Haar measure
    Mezzadri, Francesco. 2006. How to generate random matrices from the
    classical compact groups, math-ph/0609050.'''
    np.random.seed()
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q,ph,q)
    return Qobj(np.matrix(q))


def generate_training_sample(unit_nb, ctrl_init, initial, params, n_ts,evo_time,noise_name, model_dim ):
    f_ext = None
    path_template = "training/dim_{}/mtx/idx_{}"
    fid_err_targ = 1e-12
    max_iter = 200000

    max_wall_time = 5 * 60
    min_grad = 1e-20

    #target_DP = 0
    if model_dim == "2x1":
        current_path = (path_template).format(model_dim, unit_nb)
        if pathlib.Path(current_path + ".npz").exists():
            rnd_unit = Qobj(np.load(current_path + ".npz")["arr_0"])
            rnd_unitC = rnd_unit.conj()
            target_DP = tensor(rnd_unit, rnd_unitC)
        else:
            rnd_unit = tensor(rand_unitary(2),identity(2))
            rnd_unitC = rnd_unit.conj()
            np.savez(current_path, rnd_unit.full())
            target_DP = tensor(rnd_unit, rnd_unitC)
    elif model_dim == "2" or model_dim == "4":
        current_path = (path_template).format(model_dim, unit_nb)
        if pathlib.Path(current_path + ".npz").exists():
            rnd_unit = Qobj(np.load(current_path + ".npz")["arr_0"])
            rnd_unitC = rnd_unit.conj()
            target_DP = tensor(rnd_unit, rnd_unitC)
        else:
            rnd_unit = rand_unitary(int(model_dim))
            rnd_unitC = rnd_unit.conj()
            np.savez(current_path, rnd_unit.full())
            target_DP = tensor(rnd_unit, rnd_unitC)

    if noise_name == 'id_aSxbSy_spinChain_2x1':
        ctrls, drift = id_aSxbSy_spinChain_2x1(params)
    elif noise_name == "aSxbSy_id_spinChain_dim_2x1":
        ctrls, drift = aSxbSy_id_spinChain_dim_2x1(params)
    elif noise_name == "spinChainDrift_spinChain_dim_2x1":
        ctrls, drift = spinChainDrift_spinChain_dim_2x1(params)

    ctrls = [Qobj(ctrls[i]) for i in range(len(ctrls))]

    drift = Qobj(drift)

    result = cpo.optimize_pulse(drift, ctrls, initial, target_DP, n_ts, evo_time, amp_lbound=-42, amp_ubound=42,
                                fid_err_targ=fid_err_targ, min_grad=min_grad,
                                max_iter=max_iter, max_wall_time=max_wall_time,
                                out_file_ext=f_ext, init_pulse_type=ctrl_init,
                                log_level=log_level, gen_stats=True)
    print("Sample number ", unit_nb, " have error ", result.fid_err)

    np.savez("training/dim_{}/NCP_data_unbounded/idx_{}".format(model_dim,
                                                     unit_nb),result.final_amps)

if __name__ == '__main__':

    initial = identity(supeop_size)
    number_of_samples = test_set_size + train_set_size


    # pair of parameters (aplha, gamma), because our target are superoperators, then we need only NCP, so alpha=gamma=0
    params = (0., 0.)

    pathlib.Path("training/dim_{}/mtx".format(model_dim)).mkdir(parents=True, exist_ok=True)

    pathlib.Path("training/dim_{}/NCP_data_unbounded".format(model_dim)).mkdir(parents=True, exist_ok=True)

    para_generate = partial(generate_training_sample, ctrl_init=ctrl_init,
                            initial=initial, params=params,
                            n_ts=n_ts, evo_time=evo_time, noise_name=noise_name,
                            model_dim=model_dim)

    # change 8 to your number of cores
    with Pool(10) as p:
        p.map(para_generate, np.arange(12000))

    # for i in range(12000):
    #     para_generate(i)
                          
