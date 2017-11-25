import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def dense(x, n1, n2, name):
    """
    Dense fully connected layer
    :param x: Tensor, input Tensor
    :param n1: int, float, number of input neurons
    :param n2: int, float, number of output neurons
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the dense fully layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[n1, n2], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[n2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


def cnn_2d(x, weight_shape, strides, name):
    """
    A 2d Convolutional layer
    :param x: Tensor, input tensor
    :param weight_shape: Tensor, [filter_size, filter_size, n_input_features, n_output_features]
    :param strides: Tensor, [1, stride_x, stride_y, 1]
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the convolutional layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[weight_shape[-1]],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.nn.conv2d(x, filter=weights, strides=strides, padding="SAME", name="Output") + bias
        return output


def cnn_2d_trans(x, weight_shape, strides, output_shape, name):
    """
    Performs convolution transpose
    :param x: Tensor, input tensor
    :param weight_shape: Tensor, [filter_size, filter_size, n_input_features, n_output_features]
    :param strides: Tensor, [1, stride_x, stride_y, 1]
    :param output_shape: List, required output shape
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the convolutional transpose layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[weight_shape[-2]],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.nn.conv2d_transpose(x, weights, output_shape, strides=strides, padding="SAME") + bias
        return output


def conv_block(x, filter_size, stride_length, n_maps, name):
    """
    CNN block with cnn, batch norm and lrelu repeated twice. The input Tensor's H and W is down-sampled only once.
    :param x: Tensor, input tensor
    :param filter_size: int or float, size of the square kernel to use
    :param stride_length: int, stride length to be applied
    :param n_maps: int, number of output maps
    :param name: String, Name of the block on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the convolutional block
    """
    with tf.variable_scope(name):
        conv_1 = lrelu(batch_norm(cnn_2d(x, weight_shape=[filter_size, filter_size, x.get_shape()[-1], n_maps],
                                         strides=[1, stride_length, stride_length, 1], name='conv_1'), center=True,
                                  scale=True, is_training=True, scope='Batch_Norm_1'))
        conv_2 = lrelu(batch_norm(cnn_2d(conv_1, weight_shape=[filter_size, filter_size, conv_1.get_shape()[-1], n_maps],
                                         strides=[1, 1, 1, 1], name='conv_2'), center=True,
                                  scale=True, is_training=True, scope='Batch_Norm_2'))
        return conv_2


def conv_t_block(x, filter_size, stride_length, n_maps, output_shape, name):
    """
    CNN block with cnn_trans, batch norm and lrelu repeated twice. The input Tensor's H and W is up-sampled only once.
    :param x: Tensor, input tensor
    :param filter_size: int or float, size of the square kernel to use
    :param stride_length: int, stride length to be applied
    :param n_maps: int, number of output maps
    :param output_shape: String, Name of the block on Tensorboard also prefixes this value for the variable name
    :param name:
    :return: Tensor, output tensor after passing it through the convolutional transpose block
    """
    with tf.variable_scope(name):
        conv_t_1 = lrelu(batch_norm(cnn_2d_trans(x, weight_shape=[filter_size, filter_size, n_maps, x.get_shape()[-1]],
                                strides=[1, stride_length, stride_length, 1], output_shape=output_shape,
                                name='conv_t_1'), center=True, scale=True, is_training=True, scope='Batch_Norm_1'))
        conv_t_2 = lrelu(batch_norm(cnn_2d_trans(conv_t_1, weight_shape=[filter_size, filter_size, n_maps, conv_t_1.get_shape()[-1]],
                                strides=[1, 1, 1, 1], output_shape=output_shape,
                                name='conv_t_2'), center=True, scale=True, is_training=True, scope='Batch_Norm_2'))
        return conv_t_2


def lrelu(x, leak_slope=0.2):
    """
    leaky ReLU activation function
    :param x: Tensor, Input tensor
    :param leak_slope: float, slope for the negative part
    :return: Tensor, after passing it through lrelu
    """
    return tf.maximum(x, x*leak_slope)
