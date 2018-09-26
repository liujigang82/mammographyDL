import tensorflow as tf
from load-data import get_training_data, get_test_data
from create_graph import get_graph


## ARGUMENTS
epochs = 1
dataset = 10
how = "normal"
action = "eval"
threshold = 0.5
contrast = 1
weight = 7.0
distort = False

batch_size = 32

## Hyperparameters
epsilon = 1e-8

# learning rate
epochs_per_decay = 5
decay_factor = 0.80
staircase = True

# lambdas
lamC = 0.00001
lamF = 0.00250

# use dropout
dropout = True
fcdropout_rate = 0.5
convdropout_rate = 0.001
pooldropout_rate = 0.1

num_classes = 2


action = "eval"

## CONFIGURE OPTIONS
init = False
print_every = 5  # how often to print metrics
checkpoint_every = 1  # how often to save model in epochs
print_metrics = True  # whether to print or plot metrics, if False a plot will be created and updated every epoch

epochs = 1
model_name = "model_s2.0.0.36b.10"

config = tf.ConfigProto()


### read data.
train_files, total_records = get_training_data(what=dataset)
test_files, total_records = get_test_data(what=dataset)

steps_per_epoch = int(total_records / batch_size)


graph, extra_update_ops, merged, prec_op, acc_op, rec_op, recall, accuracy, mean_ce, precision, training = get_graph(dataset, decay_factor, staircase, how, batch_size, contrast, distort, lamC, epsilon, dropout, pooldropout_rate, weight, num_classes, threshold)

## train the model
with tf.Session(graph=graph, config=config) as sess:
    # create the saver
    saver = tf.train.Saver()

    # If the model is new initialize variables, else restore the session
    if init:
        sess.run(tf.global_variables_initializer())
        print("Initializing model...")
    else:
        saver.restore(sess, './model/' + model_name + '.ckpt')
        print("Restoring model", model_name)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # if we are training the model
    if action == "train":

        print("Training model", model_name, "...")

        for epoch in range(epochs):
            sess.run(tf.local_variables_initializer())

            for i in range(steps_per_epoch):
                # create the metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, precision_value, summary, acc_value, cost_value, recall_value = sess.run(
                    [extra_update_ops, prec_op, merged, accuracy, mean_ce, rec_op],
                    feed_dict={
                        training: True,
                    },
                    options=run_options,
                    run_metadata=run_metadata)

            # save checkpoint every nth epoch
            if (epoch % checkpoint_every == 0):
                print("Saving checkpoint")
                save_path = saver.save(sess, './model/' + model_name + '.ckpt')

                # Now that model is saved set init to false so we reload it next time
                init = False
    else:
        sess.run(tf.local_variables_initializer())

        # evaluate the test data
        for i in range(steps_per_epoch - 1):
            valid_acc, valid_recall, valid_precision = sess.run(
                [acc_op, rec_op, prec_op],
                feed_dict={
                    training: False
                })

        # evaluate once more to get the summary
        cv_recall, cv_precision, cv_accuracy = sess.run(
            [recall, precision, accuracy],
            feed_dict={
                training: False
            })

        print("Test Accuracy:", cv_accuracy)
        print("Test Recall:", cv_recall)
        print("Test Precision:", cv_precision)

    # stop the coordinator and the threads
    coord.request_stop()
    coord.join(threads)