"""Routine for decoding the ThaiNet dataset from TFRecords.

    Adapted from https://github.com/tensorflow/models/blob/master/tutorials/
    image/cifar10/cifar10_input.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 56

# Global constants describing the ThaiNet data set.
NUM_CLASSES = 86
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 157896
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def readThaiNet(filename_queue):
    """Reads record from ThaiNet dataset
    Args:
        filename_queue: A queue of strings with the filenames to read from.
    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (64)
            width: number of columns in the result (64)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                 for this example.
            label: an int32 Tensor with the label in the range 0..85.
            uint8image: a [height, width, depth] uint8 Tensor with image data
    """

    class ThaiNetRecord(object):
        pass
    result = ThaiNetRecord()

    result.height = 64
    result.width = 64
    result.depth = 3

    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)

    # Decode example
    features = tf.parse_single_example(
          value,
          features={
              'image_raw': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
          })

    # Decode and reshape image from string
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    result.uint8image = tf.reshape(image, [64, 64, 3])

    # Get label from features
    result.label = tf.cast(features['label'], tf.int32)

    return result


def _generateImageAndLabelBatch(image, label, min_queue_examples,
                                batchSize, shuffle):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        batchSize: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batchSize, height, width, 3] size.
        labels: Labels. 1D tensor of [batchSize] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batchSize' images + labels from the example queue.
    num_preprocess_threads = 2
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batchSize,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batchSize,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batchSize,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batchSize)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batchSize])


def distortedInputs(dataDir, batchSize):
    """Construct distorted input for ThaiNet training using the Reader ops.
    Args:
        dataDir: Path to the TFRecords file.
        batchSize: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batchSize, IMAGE_SIZE, IMAGE_SIZE, 3].
        labels: Labels. 1D tensor of [batchSize].
    """

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([dataDir])

    # Read examples from files in the filename queue.
    read_input = readThaiNet(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d ThaiNet images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generateImageAndLabelBatch(float_image, read_input.label,
                                       min_queue_examples, batchSize,
                                       shuffle=True)


def inputs(dataDir, batchSize):
    """Construct input for ThaiNet evaluation using the Reader ops.
    Args:
        evalData: bool, indicating if one uses the train or eval data set.
        dataDir: Path to the TFRecords file.
        batchSize: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batchSize, IMAGE_SIZE, IMAGE_SIZE, 3].
        labels: Labels. 1D tensor of [batchSize].
    """

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([dataDir])

    # Read examples from files in the filename queue.
    read_input = readThaiNet(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generateImageAndLabelBatch(float_image, read_input.label,
                                       min_queue_examples, batchSize,
                                       shuffle=False)
