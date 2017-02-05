import tensorflow as tf


class TrainingOp:
    def __init__(self, learning_rate, name_scope="train"):
        self.learning_rate = learning_rate
        self.name_scome = name_scope

    def add_op(self, loss, global_step):
        """
        * Sets up the training Ops.
        * Creates an optimizer and applies the gradients to all trainable variables.
        * The Op returned by this function is what must be passed to the `sess.run()` call to cause the model to train.

        :param loss: Loss tensor, from loss().
        :param global_step: Global step counter.
        :return: train_op: The Op for training.
        """

        with tf.name_scope(self.name_scome):
            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # Use the optimizer to apply the gradients that minimize the loss (and also increment the global step
            # counter) as a single training step.
            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op
