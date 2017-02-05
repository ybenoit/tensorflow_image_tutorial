from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time

import numpy as np
from six.moves import xrange
import shutil

import tensorflow as tf

from tensorflow_image_tutorial import Constant
from tensorflow_image_tutorial.input import DataLoader
from tensorflow_image_tutorial.operation import InputOp, InferenceOpWithConv, LossOp, TrainingOp, EvaluationOp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir",
                           os.path.join(os.path.dirname(__file__), "../data/NotMNIST"),
                           """Directory containing the input data""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/not_mnist/not_mnist_train/one_conv',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('summaries_dir', '/tmp/not_mnist/not_mnist_logs/one_conv', """Summaries directory""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('max_steps', 1000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('learning_rate', 0.05, """Learning Rate.""")


def main():
    """
    Train NotMNIST for a number of steps.
    """

    # Remove tensorboard previous directory
    if os.path.exists(FLAGS.summaries_dir):
        shutil.rmtree(FLAGS.summaries_dir)

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Load data
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = DataLoader(
        data_dir=FLAGS.data_dir,
        image_size=28,
        num_labels=10).load()

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        """
        Step 1 - Input data management
        """

        # Input data
        x, y = InputOp(num_pixels=Constant.IMAGE_SIZE*Constant.IMAGE_SIZE, num_classes=10).add_op()

        # Reshape images for visualization
        x_reshaped = tf.reshape(x, [-1, Constant.IMAGE_SIZE, Constant.IMAGE_SIZE, 1])
        tf.summary.image('input', x_reshaped, Constant.NUM_CLASSES)

        """
        Step 2 - Building the graph
        """

        # Build a Graph that computes the logits predictions from the inference model.
        softmax = InferenceOpWithConv(
            num_pixels=Constant.NUM_PIXELS,
            num_classes=Constant.NUM_CLASSES,
            num_neurons_in_dense_hidden_layer=100,
            num_channels_in_conv_layer=32
        ).add_ops(x_reshaped)

        # Calculate loss.
        loss = LossOp(name_scope="cross_entropy").add_op(softmax, y)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = TrainingOp(learning_rate=FLAGS.learning_rate, name_scope="train").add_op(loss, global_step)

        """
        Step 3 - Build the evaluation step
        """

        # Model Evaluation
        accuracy = EvaluationOp(name_scope="accuracy").add_op(softmax, y)

        """
        Step 4 - Merge all summaries for TensorBoard generation
        """
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Summary Writers
        train_summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        validation_summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', sess.graph)

        """
        Step 5 - Train the model, and write summaries
        """

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in xrange(FLAGS.max_steps):

            start_time = time.time()

            # Pick an offset within the training data, which has been randomized.
            offset = (step * FLAGS.batch_size) % (train_labels.shape[0] - FLAGS.batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + FLAGS.batch_size), :]
            batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]

            # Run training step and train summaries
            summary_train, _, loss_value = sess.run([summary_op, train_op, loss],
                                                    feed_dict={x: batch_data, y: batch_labels})

            train_summary_writer.add_summary(summary_train, step)

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                # Run summaries and measure accuracy on validation set
                summary_valid, acc_valid = sess.run([summary_op, accuracy],
                                                    feed_dict={x: valid_dataset, y: valid_labels})

                validation_summary_writer.add_summary(summary_valid, step)

                format_str = '%s: step %d, accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print (format_str % (datetime.now(), step, 100 * acc_valid, examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        acc_test = sess.run(accuracy, feed_dict={x: test_dataset, y: test_labels})
        print ('Accuracy on test set: %.2f' % (100 * acc_test))


if __name__ == '__main__':
    main()
