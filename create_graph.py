import tensorflow as tf

from load-data import read_and_decode_single_example, get_test_data, _scale_input_data, augment, _conv2d_batch_norm



##### Build the graph
def get_graph(dataset, decay_factor, staircase, how, batch_size, contrast, distort, lamC, epsilon, dropout, pooldropout_rate, weight, num_classes, threshold):
    graph = tf.Graph()
    model_name = "model_s2.0.0.36b.10"
    test_files, total_records = get_test_data(what=dataset)
    with graph.as_default():
        training = tf.placeholder(dtype=tf.bool, name="is_training")
        is_testing = tf.placeholder(dtype=bool, shape=(), name="is_testing")

        # create global step for decaying learning rate
        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(0.001,
                                                   global_step,
                                                   1366,
                                                   decay_factor,
                                                   staircase=staircase)

        with tf.name_scope('inputs') as scope:
            image, label = read_and_decode_single_example(test_files, label_type=how, normalize=False, distort=False)

            X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,
                                                  seed=None,
                                                  min_after_dequeue=1000)

            # Placeholders
            X = tf.placeholder_with_default(X_def, shape=[None, None, None, 1])
            y = tf.placeholder_with_default(y_def, shape=[None])

            # cast to float and scale input data
            X_adj = tf.cast(X, dtype=tf.float32)
            X_adj = _scale_input_data(X_adj, contrast=contrast, mu=127.0, scale=255.0)

            # optional online data augmentation
            if distort:
                X_adj, y = augment(X_adj, y, horizontal_flip=True, vertical_flip=True, mixup=0)

        # Convolutional layer 1
        with tf.name_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(
                X_adj,
                filters=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=100),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv1'
            )

            conv1 = tf.layers.batch_normalization(
                conv1,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn1'
            )

            # apply relu
            conv1_bn_relu = tf.nn.relu(conv1, name='relu1')

        with tf.name_scope('conv1.1') as scope:
            conv11 = tf.layers.conv2d(
                conv1_bn_relu,
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=101),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv1.1'
            )

            conv11 = tf.layers.batch_normalization(
                conv11,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn1.1'
            )

            # apply relu
            conv11 = tf.nn.relu(conv11, name='relu1.1')

        with tf.name_scope('conv1.2') as scope:
            conv12 = tf.layers.conv2d(
                conv11,
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1101),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv1.2'
            )

            conv12 = tf.layers.batch_normalization(
                conv12,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn1.2'
            )

            # apply relu
            conv12 = tf.nn.relu(conv12, name='relu1.1')

        # Max pooling layer 1
        with tf.name_scope('pool1') as scope:
            pool1 = tf.layers.max_pooling2d(
                conv12,
                pool_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name='pool1'
            )

            # optional dropout
            if dropout:
                pool1 = tf.layers.dropout(pool1, rate=pooldropout_rate, seed=103, training=training)

        # Convolutional layer 2
        with tf.name_scope('conv2.1') as scope:
            conv2 = tf.layers.conv2d(
                pool1,
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=104),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv2.1'
            )

            conv2 = tf.layers.batch_normalization(
                conv2,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn2.1'
            )

            # apply relu
            conv2 = tf.nn.relu(conv2, name='relu2.1')

        # Convolutional layer 2
        with tf.name_scope('conv2.2') as scope:
            conv22 = tf.layers.conv2d(
                conv2,
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1104),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv2.2'
            )

            conv22 = tf.layers.batch_normalization(
                conv22,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn2.2'
            )

            # apply relu
            conv22 = tf.nn.relu(conv22, name='relu2.2')

        # Max pooling layer 2
        with tf.name_scope('pool2') as scope:
            pool2 = tf.layers.max_pooling2d(
                conv22,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                name='pool2'
            )

            # optional dropout
            if dropout:
                pool2 = tf.layers.dropout(pool2, rate=pooldropout_rate, seed=106, training=training)

        # Convolutional layer 3
        with tf.name_scope('conv3.1') as scope:
            conv3 = tf.layers.conv2d(
                pool2,
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=107),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv3.1'
            )

            conv3 = tf.layers.batch_normalization(
                conv3,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn3.1'
            )

            # apply relu
            conv3 = tf.nn.relu(conv3, name='relu3.1')

        # Convolutional layer 3
        with tf.name_scope('conv3.2') as scope:
            conv32 = tf.layers.conv2d(
                conv3,
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1107),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv3.2'
            )

            conv32 = tf.layers.batch_normalization(
                conv32,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn3.2'
            )

            # apply relu
            conv32 = tf.nn.relu(conv32, name='relu3.2')

        # Max pooling layer 3
        with tf.name_scope('pool3') as scope:
            pool3 = tf.layers.max_pooling2d(
                conv32,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                name='pool3'
            )

            if dropout:
                pool3 = tf.layers.dropout(pool3, rate=pooldropout_rate, seed=109, training=training)

        # Convolutional layer 4
        with tf.name_scope('conv4') as scope:
            conv4 = tf.layers.conv2d(
                pool3,
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=110),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv4'
            )

            conv4 = tf.layers.batch_normalization(
                conv4,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn4'
            )

            # apply relu
            conv4_bn_relu = tf.nn.relu(conv4, name='relu4')

        # Max pooling layer 4
        with tf.name_scope('pool4') as scope:
            pool4 = tf.layers.max_pooling2d(
                conv4_bn_relu,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                name='pool4'
            )

            if dropout:
                pool4 = tf.layers.dropout(pool4, rate=pooldropout_rate, seed=112, training=training)

        # Convolutional layer 5
        with tf.name_scope('conv5') as scope:
            conv5 = tf.layers.conv2d(
                pool4,
                filters=512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=113),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv5'
            )

            conv5 = tf.layers.batch_normalization(
                conv5,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn5'
            )

            # apply relu
            conv5_bn_relu = tf.nn.relu(conv5, name='relu5')

        # Max pooling layer 4
        with tf.name_scope('pool5') as scope:
            pool5 = tf.layers.max_pooling2d(
                conv5_bn_relu,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                name='pool5'
            )

            if dropout:
                pool5 = tf.layers.dropout(pool5, rate=pooldropout_rate, seed=115, training=training)

        fc1 = _conv2d_batch_norm(pool5, 2048, kernel_size=(5, 5), stride=(5, 5), training=training, epsilon=1e-8,
                                 padding="VALID", seed=1013, lambd=lamC, name="fc_1")

        fc2 = _conv2d_batch_norm(fc1, 2048, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="VALID", seed=1014, lambd=lamC, name="fc_2")

        fc3 = tf.layers.dense(
            fc2,
            num_classes,  # One output unit per category
            activation=None,  # No activation function
            kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=121),
            bias_initializer=tf.zeros_initializer(),
            name="fc_logits"
        )

        logits = tf.squeeze(fc3, name="fc_flat_logits")

        # get the fully connected variables so we can only train them when retraining the network
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")

        with tf.variable_scope('conv1', reuse=True):
            conv_kernels1 = tf.get_variable('kernel')
            kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

        with tf.variable_scope('visualization'):
            tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

        # This will weight the positive examples higher so as to improve recall
        weights = tf.multiply(weight, tf.cast(tf.greater(y, 0), tf.float32)) + 1
        mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

        # Add in l2 loss
        loss = mean_ce + tf.losses.get_regularization_loss()

        # Adam optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)

        # get the probabilites for the classes
        probabilities = tf.nn.softmax(logits, name="probabilities")
        abnormal_probability = 1 - probabilities[:, 0]

        # Compute predictions from the probabilities
        if threshold == 0.5:
            predictions = tf.argmax(probabilities, axis=1, output_type=tf.int32)
        else:
            predictions = tf.cast(tf.greater(abnormal_probability, threshold), tf.int32)

        # get the accuracy
        accuracy, acc_op = tf.metrics.accuracy(
            labels=y,
            predictions=predictions,
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            name="accuracy",
        )

        recall, rec_op = tf.metrics.recall(labels=y, predictions=predictions,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                                           name="recall")
        precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS, name="precision")

        f1_score = 2 * ((precision * recall) / (precision + recall))

        # Create summary hooks
        tf.summary.scalar('accuracy', accuracy, collections=["summaries"])
        tf.summary.scalar('cross_entropy', mean_ce, collections=["summaries"])
        tf.summary.scalar('learning_rate', learning_rate, collections=["summaries"])

        # add this so that the batch norm gets run
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Merge all the summaries
        merged = tf.summary.merge_all("summaries")

        print("Graph created...")
        return graph, extra_update_ops, merged, prec_op, acc_op, rec_op, recall, accuracy, mean_ce, precision, training
