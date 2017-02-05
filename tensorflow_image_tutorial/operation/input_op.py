import tensorflow as tf


class InputOp:
    def __init__(self, num_pixels, num_classes, name_scope="input"):
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.name_scope = name_scope

    def add_op(self):
        """
        Initialises all input placeholders needed in the graph

        :return: Input and labels placeholders
        """

        with tf.name_scope(self.name_scope):

            x = tf.placeholder(tf.float32, shape=(None, self.num_pixels))
            y = tf.placeholder(tf.float32, shape=(None, self.num_classes))

        return x, y
