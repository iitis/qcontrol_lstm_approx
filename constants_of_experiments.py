import numpy as np

# Parameter testing_effectiveness controls the mode of operation of the 
# lstm_as_approximation script
# 
# It has to be set to True during the first run.
#
# For its value set to False, it performs experiment with local disturbances. In
# this case it is necessary to have a trained network. One needs to perform at
# least one experiment with value True before doing this.
testing_effectiveness = True

# size of the superoperator is dim^2
supeop_size = 16
dim = int(np.sqrt(supeop_size))

# evolution time
evo_time = 6

# number of times slots
n_ts = 32

# define appropriate model integrate_lind in noise_models_and_integration.py
# TODO: change the names of everything
# type of drift/noise
noise_name = 'id_aSxbSy_spinChain_2x1'
# noise_name = 'Sy_id_spChain'

# dimension of the target
model_dim = '2x1'
# qutip initialization of NCP
ctrl_init = 'ZERO'

# parameters depending on the available number of samples
# number of control pulses for training
train_set_size = 4000
# number of control pulses for testing
test_set_size = 1000

# parameters of tensorflow
nb_epochs = 50000
learning_rate = 0.0001
size_of_lrs = [200, 250, 350]
batch_size = 5 

# part of architecture determined by the shape of data
controls_nb = 2

# parameters controlling the noise
#gamma = 0.2
#alpha = 0.0

# there is possibility to examine performance of network for many parameters
list_gammas = [0.6,0.8]
list_alphas = [0.,0.2]
params_list = [(i, j) for i in list_gammas for j in list_gammas]

# constants for the approximation experiment
accept_err = 0.1
eps_order = 1
