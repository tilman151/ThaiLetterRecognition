import tensorflow as tf
import time
from datetime import datetime
import os
import thaiNetInput as data
import thaiNet as model

DATA_DIR = './datasetGeneration/'
TRAIN_DATA = 'train.tfrecords'
BATCH_SIZE = 32  # 150 samples per batch
LOG_FREQ = 10  # Log every 10 batches
MAX_EPOCH = 100  # 100 epochs


def train():
    """Training script for ThaiNet"""

    global_step = tf.train.get_or_create_global_step()

    maxSteps = data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * MAX_EPOCH / BATCH_SIZE

    with tf.device('/cpu:0'):
        images, labels = data.distortedInputs(
                            os.path.join(DATA_DIR, TRAIN_DATA),
                            BATCH_SIZE)

    logits = model.inference(images, trainingMode=True)

    loss = model.loss(logits, labels)

    trainOp = model.train(loss)

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)

        def after_run(self, run_context, run_values):
            if self._step % LOG_FREQ == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = LOG_FREQ * BATCH_SIZE / duration
                sec_per_batch = float(duration / LOG_FREQ)

                format_str = ('%s: step %d of %d, loss = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), self._step, maxSteps,
                       loss_value, examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=DATA_DIR,
            hooks=[tf.train.StopAtStepHook(last_step=maxSteps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()]) as sess:
        while not sess.should_stop():
            sess.run(trainOp)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
