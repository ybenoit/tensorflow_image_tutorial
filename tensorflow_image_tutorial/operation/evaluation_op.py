import tensorflow as tf


class EvaluationOp:
    def __init__(self, name_scope="accuracy"):
        self.name_scope = name_scope

    def add_op(self, logits, labels):
        """
        Adds to the inference graph the ops required to generate the model accuracy.

        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param labels: Labels tensor, int32 - [batch_size].
        :return: loss: Accuracy tensor of type float.
        """

        with tf.name_scope(self.name_scope):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar(self.name_scope, accuracy)

        return accuracy
