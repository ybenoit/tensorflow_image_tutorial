import tensorflow as tf


class InferenceOpWithThreeHidden:
    def __init__(self, num_pixels, num_classes,
                 num_neurons_in_first_dense_layer=500,
                 num_neurons_in_second_dense_layer=200,
                 num_neurons_in_third_dense_layer=50):
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.num_neurons_in_first_dense_layer = num_neurons_in_first_dense_layer
        self.num_neurons_in_second_dense_layer = num_neurons_in_second_dense_layer
        self.num_neurons_in_third_dense_layer = num_neurons_in_third_dense_layer

    def add_ops(self, x):
        """
        Build the graph as far as is required for running the network forward to make predictions.

        :param x: Images placeholder, from inputs().
        :return: softmax: Output Tensor with the computed logits.
        """
        dense_1 = self.add_dense_hidden_layer_op(x,
                                                self.num_pixels,
                                                self.num_neurons_in_first_dense_layer,
                                                name_scope="dense_1")

        dense_2 = self.add_dense_hidden_layer_op(dense_1,
                                                self.num_neurons_in_first_dense_layer,
                                                self.num_neurons_in_second_dense_layer,
                                                name_scope="dense_2")

        dense_3 = self.add_dense_hidden_layer_op(dense_2,
                                                self.num_neurons_in_second_dense_layer,
                                                self.num_neurons_in_third_dense_layer,
                                                name_scope="dense_3")

        softmax = self.add_softmax_op(dense_3, self.num_neurons_in_third_dense_layer, self.num_classes,
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

            tf.summary.histogram(softmax.op.name + '/activations', softmax)

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
