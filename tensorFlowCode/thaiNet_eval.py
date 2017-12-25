import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import math
import thaiNetInput as data
import thaiNet as model

MODEL_DIR = './model/'
DATA_DIR = './datasetGeneration/'
EVAL_DIR = './eval/'
EVAL_INTERVAL = '60*60'
BATCH_SIZE = 100
ONLY_ONCE = True


def evalOnce(topK, saver):

    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            globalStep = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,
                                                 coord=coord,
                                                 daemon=True,
                                                 start=True))

            numIter = int(math.ceil(data.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
            trueCount = 0
            totalCount = numIter * BATCH_SIZE
            step = 0
            while step < numIter and not coord.should_stop():
                predictions = sess.run([topK])
                trueCount += np.sum(predictions)
                step += 1

            # Compute accuracy
            accuracy = trueCount / totalCount
            print('%s: accuracy @ 1 = %.3f' % (datetime.now(), accuracy))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():

    with tf.Graph().as_default() as g:
        filePath = os.path.join(DATA_DIR, 'eval.tfrecords')
        images, labels = data.inputs(dataDir=filePath, batchSize=BATCH_SIZE)

        logits = model.inference(images, trainingMode=False)

        topK = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()

        while True:
            print('%s: Starting evaluation...' % datetime.now())
            evalOnce(topK, saver)
            if ONLY_ONCE:
                break
            time.sleep(EVAL_INTERVAL)


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
