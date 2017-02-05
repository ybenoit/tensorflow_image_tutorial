import tensorflow as tf


class InferenceOpWithConv:
    def __init__(self, num_pixels, num_classes, num_neurons_in_dense_hidden_layer,
                 num_channels_in_conv_layer):
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.num_neurons_in_dense_hidden_layer = num_neurons_in_dense_hidden_layer
        self.num_channels_in_conv_layer = num_channels_in_conv_layer

    def add_ops(self, x):
        """
        Build the graph as far as is required for running the network forward to make predictions.

        :param x: Images placeholder, from inputs().
        :return: softmax: Output Tensor with the computed logits.
        """
        conv = self.add_conv_op(x=x,
                                num_pixels_conv=5,
                                num_channels_in_previous_layer=1,
                                num_channels_in_current_layer=self.num_channels_in_conv_layer,
                                name_scope="conv")

        pool = self.add_max_pool_op(x=conv, name_scope="pool")

        # reshape the output from the third convolution for the fully connected layer
        shape = pool.get_shape().as_list()
        pool_reshaped = tf.reshape(pool, shape=[-1, shape[1] * shape[2] * shape[3]])

        dense = self.add_dense_hidden_layer_op(x=pool_reshaped,
                                               num_neurons_previous_layer=self.num_pixels * self.num_channels_in_conv_layer / 4,
                                               num_neurons_current_layer=self.num_neurons_in_dense_hidden_layer,
                                               name_scope="dense")

        softmax = self.add_softmax_op(x=dense,
                                      num_neurons_previous_layer=self.num_neurons_in_dense_hidden_layer,
                                      num_classes=self.num_classes,
                                      name_scope="softmax")

        return softmax

    @staticmethod
    def add_softmax_op(x, num_neurons_previous_layer, num_classes, name_scope="softmax"):
        """
        Build the softmax op to make final predictions (probabilities)

        :param x: Input Tensor
        :param num_neurons_previous_layer: Number of neurons in the previous layer of the network
        :param num_classes: Number of classes to predict
        :param name_scope: Scope Name (sub-section in TensorBoard)
        :return: softmax: Output Tensor with the computed logits.
        """

        with tf.name_scope(name_scope):
            # Model parameters
            weights = tf.Variable(tf.zeros([num_neurons_previous_layer, num_classes]))
            biases = tf.Variable(tf.zeros([num_classes]))

            softmax = tf.nn.softmax(tf.matmul(x, weights) + biases)

            tf.summary.histogram(softmax.op.name + '/activations', x)

            return softmax

    @staticmethod
    def add_dense_hidden_layer_op(x, num_neurons_previous_layer, num_neurons_current_layer, name_scope):
        """
        Build an op for a dense hidden layer

        :param x: Input Tensor
        :param num_neurons_previous_layer: Number of neurons in the previous layer of the network
        :param num_neurons_current_layer: Number of neurons in the current layer of the network
        :param name_scope: Scope Name (sub-section in TensorBoard)
        :return: Output Tensor of the hidden layer
        """
        with tf.name_scope(name_scope):
            weights = tf.Variable(tf.truncated_normal(shape=[num_neurons_previous_layer, num_neurons_current_layer],
                                                      stddev=0.1, name="weights"))
            biases = tf.Variable(tf.constant(0.1, shape=[num_neurons_current_layer], name="biases"))

            relu = tf.nn.relu(tf.matmul(x, weights) + biases, name=name_scope)

            tf.summary.histogram(relu.op.name + '/activations', relu)

            return relu

    @staticmethod
    def add_conv_op(x, num_pixels_conv, num_channels_in_previous_layer, num_channels_in_current_layer, name_scope):
        with tf.name_scope(name_scope):
            weights = tf.Variable(tf.truncated_normal(shape=[num_pixels_conv,
                                                             num_pixels_conv,
                                                             num_channels_in_previous_layer,
                                                             num_channels_in_current_layer],
                                                      stddev=0.1, name="weights"))
            biases = tf.Variable(tf.constant(0.1, shape=[num_channels_in_current_layer], name="biases"))

            relu = tf.nn.relu(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="SAME") + biases, name=name_scope)

            tf.summary.histogram(relu.op.name + '/activations', relu)

            return relu

    @staticmethod
    def add_max_pool_op(x, name_scope):
        """
        Create a max pooling step
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name_scope)
