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

def my_opt(drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        alg='GRAPE', alg_params=None,
        optim_params=None, optim_method='DEF', method_params=None,
        optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        dyn_type='GEN_MAT', dyn_params=None,
        prop_type='DEF', prop_params=None,
        fid_type='DEF', fid_params=None,
        phase_option=None, fid_err_scale_factor=None,
        tslot_type='DEF', tslot_params=None,
        amp_update_mode=None,
        init_pulse_type='DEF', init_pulse_params=None,
        pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False, init_pulse=None):

    optim = cpo.create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=min_grad,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg=alg, alg_params=alg_params, optim_params=optim_params,
        optim_method=optim_method, method_params=method_params,
        dyn_type=dyn_type, dyn_params=dyn_params,
        prop_type=prop_type, prop_params=prop_params,
        fid_type=fid_type, fid_params=fid_params,
        init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,
        pulse_scaling=pulse_scaling, pulse_offset=pulse_offset,
        ramping_pulse_type=ramping_pulse_type,
        ramping_pulse_params=ramping_pulse_params,
        log_level=log_level, gen_stats=gen_stats)

    dyn = optim.dynamics

    dyn.init_timeslots()
    # Generate initial pulses for each control
    init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])


    pgen = optim.pulse_generator
    for j in range(dyn.num_ctrls):
        init_amps[:, j] = init_pulse[:,j]

    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)

    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "dynamics generator"
        if dyn_type == 'UNIT':
            dg_name = "Hamiltonian"
        if dyn.time_depend_drift:
            msg += "Initial drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen[0])
        else:
            msg += "Drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j + 1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial state / operator:\n"
        msg += str(dyn.initial)
        msg += "\nTarget state / operator:\n"
        msg += str(dyn.target)
        logger.info(msg)

    if out_file_ext is not None:
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Initial amplitudes output to file: " + pulsefile)

    # Start the optimisation
    result = optim.run_optimization()

    if out_file_ext is not None:
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Final amplitudes output to file: " + pulsefile)

    return result


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
    max_iter = 20000

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

    ctrls = [Qobj(ctrls[i]) for i in range(len(ctrls))]

    drift = Qobj(drift)

    result = cpo.optimize_pulse(drift, ctrls, initial, target_DP, n_ts, evo_time, amp_lbound=-1, amp_ubound=1,
                                fid_err_targ=fid_err_targ, min_grad=min_grad,
                                max_iter=max_iter, max_wall_time=max_wall_time,
                                out_file_ext=f_ext, init_pulse_type=ctrl_init,
                                log_level=log_level, gen_stats=True)
    print("Sample number ", unit_nb, " have error ", result.fid_err)

    np.savez("training/dim_{}/NCP_data/idx_{}".format(model_dim,
                                                     unit_nb),result.final_amps)

if __name__ == '__main__':

    initial = identity(supeop_size)
    number_of_samples = test_set_size + train_set_size


    # pair of parameters (aplha, gamma), because our target are superoperators, then we need only NCP, so alpha=gamma=0
    params = (0., 0.)

    pathlib.Path("training/dim_{}/mtx".format(model_dim)).mkdir(parents=True, exist_ok=True)

    pathlib.Path("training/dim_{}/NCP_data".format(model_dim,
                                                   noise_name,
                                                   ctrl_init,
                                                   n_ts,
                                                   gamma)).mkdir(parents=True, exist_ok=True)

    para_generate = partial(generate_training_sample, ctrl_init=ctrl_init,
                            initial=initial, params=params,
                            n_ts=n_ts, evo_time=evo_time, noise_name=noise_name,
                            model_dim=model_dim)

    # change 8 to your number of cores
    with Pool(10) as p:
        p.map(para_generate, np.arange(number_of_samples))
                          
