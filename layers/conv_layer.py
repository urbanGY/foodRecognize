import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def conv_layer(x, kernel_shape, bias_shape, stride, padding, name):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', kernel_shape, initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
        b = tf.get_variable('biases', bias_shape, initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
        return tf.nn.relu(conv + b)
