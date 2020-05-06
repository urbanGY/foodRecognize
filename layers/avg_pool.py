import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def avg_pool(x, kernel_shape, stride, padding, name):
    return tf.nn.avg_pool(x, kernel_shape, strides=[1, stride, stride, 1], padding = padding)
