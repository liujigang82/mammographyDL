import tensorflow as tf
import os

dataPath = 'C:/Users/jigang/Downloads/ddsm-mammography'



## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()

    if label_type != 'label':
        label_type = 'label_' + label_type

    _, serialized_example = reader.read(filename_queue)
    if label_type != 'label_mask':
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'label_normal': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        # extract the data
        label = features[label_type]
        image = tf.decode_raw(features['image'], tf.uint8)

        # reshape and scale the image
        image = tf.reshape(image, [299, 299, 1])

        # random flipping of image
        if distort:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

    else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
            })

        label = tf.decode_raw(features['label'], tf.uint8)
        image = tf.decode_raw(features['image'], tf.uint8)

        label = tf.cast(label, tf.int32)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = tf.reshape(image, [288, 288, 1])
        label = tf.reshape(label, [288, 288, 1])

        # if distort:
        #     image, label = _image_random_flip(image, label)

    if normalize:
        image = tf.image.per_image_standardization(image)

    # return the image and the label
    return image, label


def get_test_data(what=10):
    test_files = os.path.join(dataPath, "training10_4.tfrecords")

    return [test_files], 11178

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def get_training_data(what=10):
    if what == 10:
        train_path_10 = os.path.join(dataPath, "training10_0.tfrecords")
        train_path_11 = os.path.join(dataPath,"training10_1.tfrecords")
        train_path_12 = os.path.join(dataPath, "training10_2.tfrecords")
        train_path_13 = os.path.join(dataPath, "training10_3.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13]
        total_records = 44712
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records


def _scale_input_data(X, contrast=None, mu=104.1353, scale=255.0):
    # if we are adjusting contrast do that
    if contrast and contrast != 1.0:
        X_adj = tf.image.adjust_contrast(X, contrast)
    else:
        X_adj = X

    # cast to float
    if X_adj.dtype != tf.float32:
        X_adj = tf.cast(X_adj, dtype=tf.float32)

    # center the pixel data
    X_adj = tf.subtract(X_adj, mu, name="centered_input")

    # scale the data
    X_adj = tf.divide(X_adj, scale)

    return X_adj

# Function to do the data augmentation on the GPU instead of the CPU, doing it on the CPU significantly slowed down training
# Taken from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
def augment(images, labels,
            horizontal_flip=False,
            vertical_flip=False,
            augment_labels=False,
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

            if augment_labels:
                labels = tf.contrib.image.transform(
                    labels,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels

def _conv2d_batch_norm(input, filters, kernel_size=(3,3), stride=(1,1), training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):
    with tf.name_scope('layer_'+name) as scope:
        conv = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
            name='conv_'+name
        )

        # apply batch normalization
        conv = tf.layers.batch_normalization(
            conv,
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
            name='bn_'+name
        )

        if activation == "relu":
            # apply relu
            conv = tf.nn.relu(conv, name='relu_'+name)
        elif activation == "elu":
            conv = tf.nn.elu(conv, name="elu_" + name)

    return conv