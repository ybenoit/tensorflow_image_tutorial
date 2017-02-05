import tensorflow as tf


class InferenceOp:
    def __init__(self, num_pixels, num_classes, name_scope="softmax"):
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.name_scope = name_scope

    def add_ops(self, x):
        """
        Build the softmax op to make final predictions (probabilities)

        :param x: Input Tensor
        :return: softmax: Output Tensor with the computed logits.
        """

        with tf.name_scope(self.name_scope):
            # Model parameters
            weights = tf.Variable(tf.zeros([self.num_pixels, self.num_classes]))
            biases = tf.Variable(tf.zeros([self.num_classes]))

            softmax = tf.nn.softmax(tf.matmul(x, weights) + biases)

        # Softmax values histogram
        tf.summary.histogram(softmax.op.name + '/activations', softmax)

        return softmax
