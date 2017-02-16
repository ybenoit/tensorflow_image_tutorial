import tensorflow as tf


class LossOp:
    def __init__(self, name_scope="cross_entropy"):
        self.name_scope = name_scope

    def add_op(self, logits, labels):
        """
        Adds to the inference graph the ops required to generate loss (cross-entropy).

        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param labels: Labels tensor, int32 - [batch_size].
        :return: loss: Loss tensor of type float.
        """

        with tf.name_scope(self.name_scope):
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name=self.name_scope)

            tf.summary.scalar(self.name_scope, cross_entropy_mean)

        return cross_entropy_mean
