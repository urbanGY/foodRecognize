import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def fc_layer(x, input_size, output_size, relu, name):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape = [input_size, output_size], initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
        b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))
        logits = tf.matmul(x, W) + b
        if relu:
            return tf.nn.relu(logits)
        else:
            return logits
