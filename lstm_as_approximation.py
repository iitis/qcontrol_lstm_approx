import os
from sys import argv, stdout
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import scipy
import scipy.io
from itertools import product as prod
import time
from tensorflow.python.client import timeline
import cProfile
from sys import argv, stdout

from get_data import *
import pathlib

from noise_models_and_integration import *
from architecture import *


# from experiments import noise_1_paramas as noise_params

def variation_acc2_local_disturb(sess,
                                 network,
                                 x_,
                                 keep_prob,
                                 saver,
                                 test_input,
                                 test_target,
                                 params):
    eps = 10 ** (-params.eps_order)

    # restoring saved model
    saver.restore(sess, "weights/dim_{}/{}/gam_{}_alfa_{}.ckpt".format(params.model_dim, params.noise_name, params.gamma, params.alpha))

    # initializoing resulting tensor, first two dimensions corresponds to coordinate which will be disturbed, on the last dimension, there will be added variation of outputs
    results = np.zeros((n_ts, controls_nb, len(np.array(test_input))))

    print(len(test_input))
    print(np.shape(results))
    iter = -1

    for sample_nb in range(len(np.array(test_input))):

        # taking sample NCP
        origin_NCP = test_input[sample_nb]
        # taking target superoperator corresponding to the NCP
        origin_superoperator = test_target[sample_nb]
        tf_result = False


        # calculating nnDCP corresponding to input NCP
        pred_DCP = get_prediction(sess, network, x_, keep_prob, np.reshape(origin_NCP, [1, params.n_ts, params.controls_nb]))
        # calculating superoperator from nnDCP
        sup_from_pred_DCP = integrate_lind(pred_DCP[0], tf_result, params)

        print("sanity check")
        acceptable_error = fidelity_err([origin_superoperator, sup_from_pred_DCP], params.dim, tf_result)
        print("predicted DCP", acceptable_error)
        print("---------------------------------")

        ############################################################################################################
        #if sanity test is above assumed error then the experiment is performed
        if acceptable_error <= params.accept_err:
            iter += 1
            # iteration over all coordinates
            for (t, c) in prod(range(params.n_ts), range(params.controls_nb)):
                new_NCP = origin_NCP
                if new_NCP[t, c] < (1 - eps):
                    new_NCP[t, c] += eps
                else:
                    new_NCP[t, c] -= eps

                sup_from_new_NCP = integrate_lind(new_NCP, tf_result, params)
                new_DCP = get_prediction(sess, network, x_, keep_prob,
                                         np.reshape(new_NCP, [1, n_ts, controls_nb]))
                sup_form_new_DCP = integrate_lind(new_DCP[0], tf_result, params)
                error = fidelity_err([sup_from_new_NCP, sup_form_new_DCP], params.dim, tf_result)

                #print(error)
                # if predicted nnDCP gives wrong superopertaor, then we add not variation of output, but some label
                if error <= params.accept_err:
                    results[t, c, iter] = np.linalg.norm(pred_DCP - new_DCP)
                else:
                    results[t, c, iter] = -1

        print(iter)

    print(np.shape(results))
    return results

def experiment_loc_disturb(params):

    ###########################################
    # PLACEHOLDERS
    ###########################################
    # input placeholder
    x_ = tf.placeholder(tf.float32, [None, params.n_ts, params.controls_nb])
    # output placeholder
    y_ = tf.placeholder(tf.complex128, [None, params.supeop_size, params.supeop_size])
    # dropout placeholder
    keep_prob = tf.placeholder(tf.float32)

    # creating the graph
    network = my_lstm(x_, keep_prob, params)

    # instance for saving the  model
    saver = tf.train.Saver()

    # loading the data
    (_, _, test_input, test_target) = get_data(params.train_set_size, params.test_set_size, params.model_dim)

    # maintaining the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # essential function which executes the experiment
        result =  variation_acc2_local_disturb(sess,
                                             network,
                                             x_,
                                             keep_prob,
                                             saver,
                                             test_input,
                                             test_target,
                                             params)

    sess.close()
    tf.reset_default_graph()
    return result


def train_and_predict(params, file_name):

    ###########################################
    # PLACEHOLDERS
    ###########################################
    # input placeholder
    x_ = tf.placeholder(tf.float32, [None, params.n_ts, params.controls_nb])
    # output placeholder
    y_ = tf.placeholder(tf.complex128, [None, params.supeop_size, params.supeop_size])
    # dropout placeholder
    keep_prob = tf.placeholder(tf.float32)

    # creating the graph
    network = my_lstm(x_, keep_prob, params)

    # instance for saving the  model
    saver = tf.train.Saver()

    # loading the data
    (train_input, train_target, test_input, test_target) = get_data(params.train_set_size,
                                                                    params.test_set_size,
                                                                    params.model_dim,
                                                                    params.data_type)
    # maintaining the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # training the network
        (acc,train_table,test_table) = fit(sess,
                  network,
                  x_,
                  y_,
                  keep_prob,
                  train_input,
                  train_target,
                  test_input,
                  test_target,
                  params)

        # making prediction by trained model
        pred = get_prediction(sess, network, x_, keep_prob, test_input)
        # saving trained model
        saver.save(sess, "weights/weights_from_{}.ckpt".format(file_name))

        sess.close()
    tf.reset_default_graph()
    return (pred,acc,train_table,test_table)


# ---------------------------------------------------------------------------

def main(testing_effectiveness,argv_number):
    config_path = "configurations/"
    file_name = 'config{}.txt'.format(argv_number)
    file = open(config_path+file_name, "r")
    parameters = dict_to_ntuple(eval(file.read()), "parameters")

    print(parameters.activ_fn)

    pathlib.Path("weights/dim_{}/{}".format(parameters.model_dim, parameters.noise_name)).mkdir(parents=True, exist_ok=True)


    if testing_effectiveness:
        pathlib.Path("results/prediction/dim_{}".format(parameters.model_dim)).mkdir(parents=True, exist_ok=True)
        # main functionality
        if os.path.isfile("results/eff_fid_lstm/experiment_{}".format(file_name[0:-4])+".npz"):
            statistic = list(np.load("results/eff_fid_lstm/experiment_{}".format(file_name[0:-4])+".npz")["arr_0"][()])
        else:
            statistic = []

        for i in range(5):
            pred, acc, train_table, test_table = train_and_predict(parameters,file_name)


            # statistic.append(acc)
            statistic.append(pred)
            # save the results
            print(acc)

            # np.savez("results/eff_fid_lstm/experiment_{}".format(file_name[0:-4]), statistic)
            np.savez("results/prediction/experiment_{}".format(file_name[0:-4]), statistic)


    else:

         # main functionality
         data = experiment_loc_disturb(n_ts,
                                      gamma,
                                      alpha,
                                      evo_time,
                                      supeop_size,
                                      controls_nb,
                                      train_set_size,
                                      test_set_size,
                                      size_of_lrs,
                                      noise_name,
                                      model_dim,
                                      eps,
                                      accept_err)

         pathlib.Path("results/NN_as_approx/dim_{}".format(model_dim)).mkdir(parents=True, exist_ok=True)
         np.savez("results/NN_as_approx/experiment_{}".format(file_name[0:-4]), data)

    file.close()


if __name__ == "__main__":

    # prepare dirs for the output files


    # Note: change the below value if you have already trained the network
    # train_model = True
    if len(argv) == 2:
        argv_number = int(argv[1])
    else:
        argv_number = 63
    main(True,argv_number )



