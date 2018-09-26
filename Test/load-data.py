import os
import numpy as np
import tensorflow as tf


dataPath = 'C:/Users/jigang/Downloads/ddsm-mammography'

epochs = 1


def get_training_data():
    train_path_10 = os.path.join(dataPath, "training10_0.tfrecords")
    return [train_path_10]

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


trainData = get_training_data()


image, label = read_and_decode_single_example(trainData, label_type="normal", normalize=False, distort=False)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  image_array = image.eval()
  print(image_array.shape)

print("done")

'''
reader = tf.TFRecordReader()
_, serialized_example = reader.read(trainData)

features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'label_normal': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    })

# extract the data
label = features['label_normal']
image = tf.decode_raw(features['image'], tf.uint8)

# reshape and scale the image
image = tf.reshape(image, [299, 299, 1])
'''