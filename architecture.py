import tensorflow as tf
from functools import partial
from sys import stdout
from sklearn.model_selection import KFold
import time
from tensorflow.python.client import timeline

from noise_models_and_integration import *

def fidelity_cost_fn(network,y_, params):

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params )
    net = tf.cast(network, tf.complex128)
    ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices

    # batch_to_loss_fn = tf.stack([y_, ctrls_to_mtx], axis=1)  # create tensor of pairs (target, generated_matrix)
    tmp_fid_err = partial(fidelity_err, dim=params.dim, tf_result=True)
    batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)  # batch of fidelity errors

    loss = tf.cast(tf.reduce_mean(batch_of_fid_err),
                   tf.float32)  # loss function, which is a mean of fid_erros over batch
    tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)
    return (optimizer, accuracy)

random_drift = [[[np.random.normal(0,0.1) for i in range(2)]for j in range(32)] for k in range(100)]

def fidelity_fixed_averaged_drift_cost(network,y_, params):
    cost_params = dict_to_ntuple(params.cost, "costParams")

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params )
    tmp_fid_err = partial(fidelity_err, dim=params.dim, tf_result=True)

    tmp = []
    for i in range(cost_params.nb_of_smpls):
        net = tf.cast(network + tf.constant(random_drift[i]),tf.complex128)
        ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices
        batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)  # batch of fidelity errors
        tmp.append(batch_of_fid_err)

    loss = tf.cast(tf.reduce_mean(tmp),tf.float32)  # loss function, which is a mean of fid_erros over batch
    tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    # accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)

    return optimizer

def fidelity_averaged_drift_cost(network,y_, params):
    cost_params = dict_to_ntuple(params.cost, "costParams")

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params )
    tmp_fid_err = partial(fidelity_err, dim=params.dim, tf_result=True)

    flucts = [tf.random_normal(tf.shape(network), mean=0.0, stddev=cost_params.fluct_std,
                               seed=np.random.seed()) for i in range(cost_params.nb_of_smpls)]

    def fidelity_averaged_drift_cost_HELPER(fluct):
        # fluct = tf.random_normal([params.n_ts, params.controls_nb], mean=0.0, stddev=cost_params.fluct_std,
        #                          seed=np.random.seed())
        net = tf.cast(network + fluct, tf.complex128)
        ctrls_to_mtx = tf.map_fn(tmp_integrate_lind,
                                 net)  # new batch in which instead of control pulses i have matrices
        batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)
        return batch_of_fid_err

    # tmp = []
    # for i in range(cost_params.nb_of_smpls):
    #     fluct = tf.random_normal(tf.shape(network), mean=0.0, stddev=cost_params.fluct_std,
    #                              seed=np.random.seed())
    #     net = tf.cast(network + fluct, tf.complex128)
    #     ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices
    #     batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)  # batch of fidelity errors
    #     tmp.append(batch_of_fid_err)

    tmp = tf.map_fn(fidelity_averaged_drift_cost_HELPER, tf.convert_to_tensor(flucts))

    loss = tf.cast(tf.reduce_mean(tmp),tf.float32)  # loss function, which is a mean of fid_erros over batch
    # tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    # accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)

    return optimizer



def fidelity_averaged_drift_fixed_per_batch_cost(network,y_, params):
    cost_params = dict_to_ntuple(params.cost, "costParams")
    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params )
    tmp_fid_err = partial(fidelity_err, dim=params.dim, tf_result=True)

    flucts = [tf.random_normal([params.n_ts, params.controls_nb], mean=0.0, stddev=cost_params.fluct_std,
                                 seed=np.random.seed()) for i in range(cost_params.nb_of_smpls) ]

    def fidelity_averaged_drift_fixed_per_batch_cost_HELPER(fluct):
        # fluct = tf.random_normal([params.n_ts, params.controls_nb], mean=0.0, stddev=cost_params.fluct_std,
        #                          seed=np.random.seed())
        net = tf.cast(network + fluct, tf.complex128)
        ctrls_to_mtx = tf.map_fn(tmp_integrate_lind,
                                 net)  # new batch in which instead of control pulses i have matrices
        batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)
        return batch_of_fid_err

    # tmp = []
    # for i in range(cost_params.nb_of_smpls):
    #     fluct = tf.random_normal([params.n_ts,params.controls_nb], mean=0.0, stddev=cost_params.fluct_std,
    #                              seed=np.random.seed())
    #     net = tf.cast(network + fluct, tf.complex128)
    #     ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices
    #     batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)  # batch of fidelity errors
    #     tmp.append(batch_of_fid_err)
    # tmp = partial(fidelity_averaged_drift_fixed_per_batch_cost_HELPER,params=params,cost_params=cost_params,y_=y_,network=network)
    tmp = tf.map_fn(fidelity_averaged_drift_fixed_per_batch_cost_HELPER,tf.convert_to_tensor(flucts))
    print(tmp)

    loss = tf.cast(tf.reduce_mean(tmp),tf.float32)  # loss function, which is a mean of fid_erros over batch
    # tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    # accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)

    return optimizer

def fidelity_accuracy(network,y_, params):

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params )
    tmp_fid_err = partial(fidelity_err, dim=params.dim, tf_result=True)


    net = tf.cast(network, tf.complex128)
    ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices
    batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx], dtype=tf.float32)  # batch of fidelity errors

    accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)

    return accuracy


def fidelity_L1_cost(network,y_, params):
    cost = dict_to_ntuple(params.cost, "cost")

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params)
    net = tf.cast(network, tf.complex128)

    ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices

    # batch_to_loss_fn = tf.stack([y_, ctrls_to_mtx], axis=1)  # create tensor of pairs (target, generated_matrix)
    tmp_fid_err = partial(fidelity_with_minL1,dim=params.dim, n_ts=params.n_ts, mi=cost.mi, b=cost.b)

    batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx, tf.cast(network,tf.float64)], dtype=tf.float32)  # batch of fidelity errors

    loss = tf.cast(tf.reduce_mean(batch_of_fid_err),
                   tf.float32)  # loss function, which is a mean of fid_erros over batch
    # tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)
    return optimizer

def fidelity_low_pass_cost(network,y_, params):
    cost = dict_to_ntuple(params.cost, "cost")

    tmp_integrate_lind = partial(integrate_lind, tf_result=True, params=params)
    net = tf.cast(network, tf.complex128)

    ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices

    # batch_to_loss_fn = tf.stack([y_, ctrls_to_mtx], axis=1)  # create tensor of pairs (target, generated_matrix)
    tmp_fid_err = partial(fidelity_with_low_pass,dim=params.dim, n_ts=params.n_ts, mi=cost.mi, delta=cost.delta)

    batch_of_fid_err = tf.map_fn(tmp_fid_err, [y_, ctrls_to_mtx, tf.cast(network,tf.float64)], dtype=tf.float32)  # batch of fidelity errors

    loss = tf.cast(tf.reduce_mean(batch_of_fid_err),
                   tf.float32)  # loss function, which is a mean of fid_erros over batch
    # tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)
    # accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)
    return optimizer






def my_lstm(x_, keep_prob, params):
    # 'layers' is a list of the number of the units on each layer

    cells = []
    for n_units in params.size_of_lrs:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=n_units, use_peepholes=True)
        # cell = tf.nn.rnn_cell.GRUCell(num_units=n_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cells.append(cell)

    outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells,
        cells_bw=cells,
        inputs=x_,
        dtype=tf.float32,
        parallel_iterations=32
    )


    # print(outputs[2])
    output_fw, output_bw= tf.split(outputs[0], 2, axis=2)
    sum_fw_bw = tf.add(output_fw, output_bw)
    squeezed_layer = tf.reshape(sum_fw_bw, [-1, params.size_of_lrs[-1]])
    droput = tf.nn.dropout(squeezed_layer, keep_prob)
    dense = tf.contrib.layers.fully_connected(droput, params.controls_nb, activation_fn=params.activ_fn)
    output = tf.reshape(dense, [tf.shape(x_)[0],tf.shape(x_)[1], params.controls_nb])
    return output



def fit(sess,
      network,
      x_,
      y_,
      keep_prob,
      train_input,
      train_target,
      test_input,
      test_target,
      params):

    # tensorboard_path = 'tensorboard/' + str(time.ctime())

    if params.cost["fn"] == "superop_fidel_L1":
        optimizer = fidelity_L1_cost(network, y_, params)
        _, accuracy = fidelity_cost_fn(network,y_, params)
    elif params.cost["fn"] == "fidelity_low_pass_cost":
        optimizer = fidelity_low_pass_cost(network, y_, params)
        _, accuracy = fidelity_cost_fn(network, y_, params)
    elif params.cost["fn"] == "superop_fidel":
        optimizer, accuracy = fidelity_cost_fn(network, y_, params)
    elif params.cost["fn"] == "fidelity_averaged_drift_cost":
        optimizer = fidelity_averaged_drift_cost(network, y_, params)
        accuracy = fidelity_accuracy(network, y_, params)
    elif params.cost["fn"] == "fidelity_averaged_drift_fixed_per_batch_cost":
        optimizer = fidelity_averaged_drift_fixed_per_batch_cost(network, y_, params)
        accuracy = fidelity_accuracy(network, y_, params)
    elif params.cost["fn"] == "fidelity_fixed_averaged_drift_cost":
        optimizer = fidelity_fixed_averaged_drift_cost(network, y_, params)
        accuracy = fidelity_accuracy(network, y_, params)

    # 500 is the number of test samples used in monitoring the efficiency of the network
    test_sample_indices = np.arange(500)
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    kf = KFold(n_splits=(params.train_set_size//params.batch_size), shuffle = True)
    print(np.shape(test_input))
    # LEARNING LOOP
    with sess.as_default():

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        j = -1
        train_table = []
        test_table = []
        for i in range(int(np.ceil(params.nb_epochs / (params.train_set_size // params.batch_size)))):
            for train_index, rand in kf.split(train_input, train_target):
                j += 1
                batch = (train_input[rand], train_target[rand])
                # batch = (train_input[(j%train_set_size):((j+batch_size)%train_set_size)], train_target[(j%train_set_size):((j+batch_size)%train_set_size)])
                # MONITORING OF EFFICENCY
                if j % 1000 == 0:
                    train_accuracy = sess.run( accuracy, feed_dict={x_: batch[0],
                                                                    y_: batch[1],
                                                                    keep_prob: 1.0})
                    train_table.append(train_accuracy)

                    test_accuracy = accuracy.eval(feed_dict={x_: test_input[test_sample_indices],
                                                             y_: test_target[test_sample_indices],
                                                             keep_prob: 1.0})
                    # test_table.append(test_accuracy)
                    # train_writer.add_summary(summary, j)
                    print("step %d, training accuracy %g" % (j, train_accuracy))
                    stdout.flush()
                    print("step %d, test accuracies %g" % (j, test_accuracy))
                    stdout.flush()
                    print (" ")
                    stdout.flush()
                sess.run(optimizer,
                         feed_dict={x_: batch[0],
                                    y_: batch[1],
                                    keep_prob: params.keep_drop})#,options=options, run_metadata=run_metadata)

                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #
                # with open('timeline_02_step_{}.json'.format(i), 'w') as f:
                #     f.write(chrome_trace)
        test_accuracy = accuracy.eval(feed_dict={x_: test_input,
                                                 y_: test_target,
                                                 keep_prob: 1.})

    return (test_accuracy,train_table,test_table)


def get_prediction(sess, network, x_, keep_prob, test_input):

    prediction = sess.run(network, feed_dict={x_:test_input,
                                              keep_prob: 1.0})
    return prediction
