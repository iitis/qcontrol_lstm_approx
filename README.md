# qcontrol_lstm_approx

# Steps required to execute the code

* Run `generate_data.py` to generate random unitary matrices and control pulses
  for learning
* Run `lstm_as_approximation.py` to use generated data for do de experiments i.e. train network and note effectivenes and 
  test the local disturbances.
* Run `printing_results.ipynb` to plot results of experiments.



# Description of file

* `generate_data.py` - generate data and create directory structure
* `get_data.py` - functions for loading data from files
* `constants_of_experiments.py` - this is control panel, with all needed parameters of experiments.
* `architecture.py` - LSTM architecture and cost functions
* `noise_models_and_integration.py` - models of quantum systems and related
  functions
* `lstm_as_approximation.py` - the main file with experiments.
* `printing_results.ipynb` - file divided into blocks in which we can plot results of experiment.
