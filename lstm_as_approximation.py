import os
from sys import argv, stdout
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import scipy
import scipy.io
from itertools import product as prod

from tensorflow.python.client import timeline

from get_data import *
import pathlib
import time
from noise_models_and_integration import *
from architecture import *
from constants_of_experiments import *


def variation_acc2_local_disturb(sess,
                                 network,
                                 x_,
                                 keep_prob,
                                 saver,
                                 noise_name,
                                 gamma,
                                 alpha,
                                 controls_nb,
                                 test_input,
                                 test_target,
                                 n_ts,
                                 evo_time,
                                 eps,
                                 accept_err):

    # restoring saved model
    saver.restore(sess, "weights/dim_{}/{}/gam_{}_alfa_{}.ckpt".format(model_dim, noise_name, gamma, alpha))

    # initializoing resulting tensor, first two dimensions corresponds to coordinate which will be disturbed, on the last dimension, there will be added variation of outputs
    results = np.zeros((n_ts, controls_nb, len(np.array(test_input))))

    print(len(test_input))
    print(np.shape(results))
    iter = -1
    tf_result = False
    for sample_nb in range(len(np.array(test_input))):

        # taking sample NCP
        origin_NCP = test_input[sample_nb]
        # origin_NCP = np.asarray(list(zip(origin_NCP[:,1],origin_NCP[:,0])))[::-1]
        # taking target superoperator corresponding to the NCP
        origin_superoperator = test_target[sample_nb]
        # origin_superoperator = integrate_lind(origin_NCP, (0., 0.), n_ts, evo_time, noise_name, tf_result)



        # calculating nnDCP corresponding to input NCP
        pred_DCP = get_prediction(sess, network, x_, keep_prob, np.reshape(origin_NCP, [1, n_ts, controls_nb]))
        # calculating superoperator from nnDCP
        sup_from_pred_DCP = integrate_lind(pred_DCP[0], (gamma, alpha), n_ts, evo_time, noise_name, tf_result)

        print("sanity check")
        error_of_DCP= fidelity_err([origin_superoperator, sup_from_pred_DCP], dim, tf_result)
        print("predicted DCP", error_of_DCP)
        print("---------------------------------")

        ############################################################################################################
        #if sanity test is above assumed error then the experiment is performed
        if error_of_DCP <= accept_err:
            iter += 1
            # iteration over all coordinates
            for (t, c) in prod(range(n_ts), range(controls_nb)):
                new_NCP = origin_NCP
                if new_NCP[t, c] < (1 - eps):
                    new_NCP[t, c] += eps
                else:
                    new_NCP[t, c] -= eps

                sup_from_new_NCP = integrate_lind(new_NCP, (0., 0.), n_ts, evo_time, noise_name, tf_result)
                new_DCP = get_prediction(sess, network, x_, keep_prob,
                                         np.reshape(new_NCP, [1, n_ts, controls_nb]))

                sup_form_new_DCP = integrate_lind(new_DCP[0], (gamma, alpha), n_ts, evo_time, noise_name, tf_result)

                error = fidelity_err([sup_from_new_NCP, sup_form_new_DCP], dim, tf_result)

                #print(error)
                # if predicted nnDCP gives wrong superopertaor, then we add not variation of output, but some label
                if error <= accept_err:
                    results[t, c, iter] = np.linalg.norm(pred_DCP - new_DCP)
                else:
                    results[t, c, iter] = -1

        print(iter)

    print(np.shape(results))
    return results


def experiment_loc_disturb(n_ts,
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
                          accept_err):
    ###########################################
    # PLACEHOLDERS
    ###########################################
    # input placeholder
    x_ = tf.placeholder(tf.float32, [None, n_ts, controls_nb])
    # output placeholder
    y_ = tf.placeholder(tf.complex128, [None, supeop_size, supeop_size])
    # dropout placeholder
    keep_prob = tf.placeholder(tf.float32)

    # creating the graph
    network = my_lstm(x_, controls_nb, size_of_lrs, keep_prob)

    # instance for saving the  model
    saver = tf.train.Saver()

    # loading the data
    (_, _, test_input, test_target) = get_data(train_set_size, test_set_size, model_dim,data_type)

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
                                             noise_name,
                                             gamma,
                                             alpha,
                                             controls_nb,
                                             test_input,
                                             test_target,
                                             n_ts,
                                             evo_time,
                                             eps,
                                             accept_err)

    sess.close()
    tf.reset_default_graph()
    return result


def train_and_predict(n_ts,
                      model_params,
                      evo_time,
                      batch_size,
                      supeop_size,
                      controls_nb,
                      nb_epochs,
                      learning_rate,
                      train_set_size,
                      test_set_size,
                      size_of_lrs,
                      dim,
                      noise_name,
                      model_dim):

    ###########################################
    # PLACEHOLDERS
    ###########################################
    # input placeholder
    x_ = tf.placeholder(tf.float32, [None, n_ts, controls_nb])
    # output placeholder
    y_ = tf.placeholder(tf.complex128, [None, supeop_size, supeop_size])
    # dropout placeholder
    keep_prob = tf.placeholder(tf.float32)

    # creating the graph
    network = my_lstm(x_,controls_nb, size_of_lrs, keep_prob)

    # instance for saving the  model
    saver = tf.train.Saver()

    # loading the data
    (train_input, train_target, test_input, test_target) = get_data(train_set_size,
                                                                    test_set_size,
                                                                    model_dim,data_type)




    # maintaining the memory
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44

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
                  nb_epochs,
                  batch_size,
                  train_set_size,
                  learning_rate,
                  model_params,
                  n_ts,
                  evo_time,
                  dim,
                  noise_name)

        # making prediction by trained model
        pred = get_prediction(sess, network, x_, keep_prob, test_input)
        # saving trained model
        if noise_name == "spinChainDrift_spinChain_dim_2x1":
            gamma, alpha, beta = model_params
            saver.save(sess, "weights/dim_{}/{}/gam_{}_alfa_{}_beta_{}.ckpt".format(model_dim, noise_name, gamma, alpha,beta))
        else:
            gamma, alpha= model_params
            saver.save(sess, "weights/dim_{}/{}/gam_{}_alfa_{}.ckpt".format(model_dim, noise_name, gamma, alpha))

        sess.close()
    tf.reset_default_graph()
    return (pred,acc,train_table,test_table)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # time.sleep(3600 * 15)
    # prepare dirs for the output files
    pathlib.Path("weights/dim_{}/{}".format(model_dim, noise_name)).mkdir(parents=True, exist_ok=True)

    # Note: change the below value if you have already trained the network
    # train_model = True


    if testing_effectiveness:
        pathlib.Path("results/eff_fid_lstm_unbounded/dim_{}".format(model_dim)).mkdir(parents=True, exist_ok=True)
        # main functionality
        statistic = dict()
        for i in range(1):
            pred, acc, train_table, test_table = train_and_predict(n_ts,
                                      model_params,
                                      evo_time,
                                      batch_size,
                                      supeop_size,
                                      controls_nb,
                                      nb_epochs,
                                      learning_rate,
                                      train_set_size,
                                      test_set_size,
                                      size_of_lrs,
                                      dim,
                                      noise_name,
                                      model_dim)


            statistic[i] = (acc, train_table, test_table)
            # save the results
            if noise_name == "spinChainDrift_spinChain_dim_2x1":
                # gamma, alpha, beta = model_params
                np.savez("results/eff_fid_lstm/dim_{}/statistic_{}_gam_{}_alpha_{}_beta_{}".format(model_dim,
                                                                                           noise_name,
                                                                                           gamma,
                                                                                           alpha, beta), statistic)
            else:
                # gamma, alpha = model_params
                np.savez("results/eff_fid_lstm_undbounded/dim_{}/statistic_{}_gam_{}_alpha_{}".format(model_dim,
                                                                                           noise_name,
                                                                                           gamma,
                                                                                           alpha), statistic)

    else:

         eps = 10**(-eps_order)


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
         np.savez("results/NN_as_approx/dim_{}/{}_gam_{}_alpha_{}_epsilon_1e-{}".format(model_dim,
                                                                                        noise_name,
                                                                                        gamma,
                                                                                        alpha,
                                                                                        eps_order), data)

