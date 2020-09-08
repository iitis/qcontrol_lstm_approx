# qcontrol_lstm_approx

# Steps required to execute the code

* Run `generate_data.py` to generate random unitary matrices and control pulses
  for learning
* Set `testing_effectiveness` to `True` in the argument of the `main()` function.
* Run `lstm_as_approximation.py` to use generated matrices to train the network and test its efficiency 
* Set `testing_effectiveness` to `False` in the argument of the `main()` function.
* Run `lstm_as_approximation.py` to use the trained network for testing the effect of local disturbances.
* Run `printing_results.ipynb` to plot results of the experiments.

# Description of file

* `generate_data.py` - generate data and create directory structure
* `get_data.py` - functions for loading data from files
* `configuratons/` - this is folder with control panels, with all needed parameters of experiments.
* `architecture.py` - LSTM architecture and cost functions
* `noise_models_and_integration.py` - models of quantum systems and related
  functions
* `lstm_as_approximation.py` - the main file with experiments.
* `printing_results.ipynb` - file divided into blocks in which we can plot results of experiment.

# Requirements

The code has been tested with Python 3.6 distributed with Anaconda. The packages
utilizes QuTIP is based on TensorFlow library. It also utilizes QuTip for generating
control pulses.

# Citing

M. Ostaszewski, J.A. Miszczak, P. Sadowski, L. Banchi,  *Approximation of quantum control correction scheme using deep neural networks*, https://arxiv.org/abs/1803.05193, Quantum Inf Process (2019), 18:126, https://doi.org/10.1007/s11128-019-2240-7
