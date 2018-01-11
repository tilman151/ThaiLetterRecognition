import tensorflow as tf
import time
from datetime import datetime
import os
import shutil
import thaiNetInput as data
import thaiNet as model

DATA_DIR = './datasetGeneration/'
MODEL_DIR = './model/'
TRAIN_DATA = 'train.tfrecords'
BATCH_SIZE = 100  # 150 samples per batch
LOG_FREQ = 10  # Log every 10 batches
SAVE_FREQ = 100  # Saves a checkpoint every 100 batches
MAX_EPOCH = 100  # 100 epochs


def train():
    """Training script for ThaiNet"""

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.mkdir(MODEL_DIR)

    global_step = tf.train.get_or_create_global_step()

    maxSteps = data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * MAX_EPOCH / BATCH_SIZE

    with tf.device('/cpu:0'):
        images, labels = data.distortedInputs(
                            os.path.join(DATA_DIR, TRAIN_DATA),
                            BATCH_SIZE)

    logits = model.inference(images, trainingMode=True)

    loss = model.loss(logits, labels)

    trainOp = model.train(loss)
    
    saver = tf.train.Saver(max_to_keep=4)
    modelFile = os.path.join(MODEL_DIR, 'ThaiNetFinal')

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

        def end(self, sess):
            saver.save(sess, modelFile, global_step=self._step)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir='./ckpts/',
            hooks=[tf.train.StopAtStepHook(last_step=maxSteps),
                   tf.train.NanTensorHook(loss),
                   tf.train.CheckpointSaverHook(checkpoint_dir=MODEL_DIR,
                                                save_steps=SAVE_FREQ,
                                                saver=saver,
                                                checkpoint_basename='ThaiNet'),
                   _LoggerHook()]) as sess:
        while not sess.should_stop():
            sess.run(trainOp)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
