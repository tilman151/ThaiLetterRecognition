import tensorflow as tf

def inference(images, trainingMode):
    """Infers the class of the image"""

    conv1 = tf.layers.conv2d(
                inputs=images,
                filters=10,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu,
                name='conv1')

    pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[2, 2],
                strides=2,
                name='pool1')

    conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=20,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu,
                name='conv2')

    pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=[2, 2],
                strides=2,
                name='pool2')

    conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=150,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu,
                name='conv3')

    conv3Flat = tf.reshape(conv3, [-1, 14 * 14 * 150])
    fc1 = tf.layers.dense(
                inputs=conv3Flat,
                units = 400,
                activation=tf.nn.relu,
                name='fc1')

    dropout1 = tf.layers.dropout(
                inputs=fc1,
                rate=0.4,
                training=trainingMode,
                name='dropout1')

    fc2 = tf.layers.dense(
                inputs=dropout1,
                units=86,
                name='fc2')

    return fc2

def loss(prediction, labels):
    """Calculates the loss between prediction and labels"""

    labels = tf.cast(labels, tf.int64)
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=prediction,
                    name='crossEntropy')
    meanEntropy = tf.reduce_mean(crossEntropy, name='meanEntropy')

    return meanEntropy

def train(totalLoss):
    """Trains the network by optimizing the loss"""

    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(totalLoss)

    applyGradient = optimizer.apply_gradients(
                        grads,
                        global_step=tf.train.get_global_step())

    return applyGradient
