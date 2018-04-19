import tensorflow as tf
from functools import partial
from sklearn.model_selection import KFold

from tensorflow.python.client import timeline

from noise_models_and_integration import *

def fidelity_cost_fn(network,y_, learning_rate, params, n_ts, evo_time,dim, noise_name):

    tmp_integrate_lind = partial(integrate_lind, params=params, n_ts=n_ts, evo_time=evo_time, noise_name=noise_name, tf_result=True)
    net = tf.cast(network, tf.complex128)

    ctrls_to_mtx = tf.map_fn(tmp_integrate_lind, net)  # new batch in which instead of control pulses i have matrices

    batch_to_loss_fn = tf.stack([y_, ctrls_to_mtx], axis=1)  # create tensor of pairs (target, generated_matrix)
    tmp_fid_err = partial(fidelity_err, dim=dim, tf_result=True)
    batch_of_fid_err = tf.map_fn(tmp_fid_err, batch_to_loss_fn, dtype=tf.float32)  # batch of fidelity errors

    loss = tf.cast(tf.reduce_mean(batch_of_fid_err),
                   tf.float32)  # loss function, which is a mean of fid_erros over batch
    tf.summary.scalar('loss_func', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    accuracy = tf.cast(tf.reduce_mean(1 - batch_of_fid_err), tf.float32)
    return (optimizer, accuracy)

def my_lstm(x_,controls_nb, size_of_lrs, keep_prob):
    # 'layers' is a list of the number of the units on each layer

    cells = []
    for n_units in size_of_lrs:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=n_units, use_peepholes=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cells.append(cell)

    outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells,
        cells_bw=cells,
        inputs=x_,
        dtype=tf.float32,
        parallel_iterations=32
    )

    output_fw, output_bw= tf.split(outputs[0], 2, axis=2)
    sum_fw_bw = tf.add(output_fw, output_bw)
    squeezed_layer = tf.reshape(sum_fw_bw, [-1, size_of_lrs[-1]])
    droput = tf.nn.dropout(squeezed_layer, keep_prob)
    dense = tf.contrib.layers.fully_connected(droput, controls_nb, activation_fn=tf.nn.tanh)
    output = tf.reshape(dense, [tf.shape(x_)[0],tf.shape(x_)[1], controls_nb])
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
      nb_epochs,
      batch_size,
      train_set_size,
      learning_rate,
      gamma,
      alpha,
      n_ts,
      evo_time,
      dim,
      noise_name):

    params = (alpha,gamma)
    optimizer, accuracy = fidelity_cost_fn(network, y_, learning_rate, params, n_ts, evo_time,dim, noise_name)


    # 500 is the number of test samples used in monitoring the efficiency of the network
    test_sample_indices = np.arange(500)

    kf = KFold(n_splits=(train_set_size//batch_size), shuffle = True)
    print(np.shape(test_input))
    # LEARNING LOOP
    with sess.as_default():

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        j = -1
        for i in range(int(np.ceil(nb_epochs / (train_set_size // batch_size)))):
            for train_index, rand in kf.split(train_input, train_target):
                j += 1
                batch = (train_input[rand], train_target[rand])
                # MONITORING OF EFFICENCY
                if j % 1000 == 0:
                    train_accuracy = sess.run( accuracy, feed_dict={x_: batch[0],
                                                                    y_: batch[1],
                                                                    keep_prob: 1.0})

                    test_accuracy = accuracy.eval(feed_dict={x_: test_input[test_sample_indices],
                                                             y_: test_target[test_sample_indices],
                                                             keep_prob: 1.0})

                    print("step %d, training accuracy %g" % (j, train_accuracy))
                    print("step %d, test accuracies %g" % (j, test_accuracy))
                    print (" ")
                sess.run(optimizer,
                         feed_dict={x_: batch[0],
                                    y_: batch[1],
                                    keep_prob: 0.5})#,options=options, run_metadata=run_metadata)

                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #
                # with open('timeline_02_step_{}.json'.format(i), 'w') as f:
                #     f.write(chrome_trace)
        test_accuracy = accuracy.eval(feed_dict={x_: test_input,
                                                 y_: test_target,
                                                 keep_prob: 1.})

    return test_accuracy


def get_prediction(sess, network, x_, keep_prob, test_input):

    prediction = sess.run(network, feed_dict={x_:test_input,
                                              keep_prob: 1.0})
    return prediction
