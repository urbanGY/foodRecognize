import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from layers.conv_layer import conv_layer
from layers.max_pool import max_pool

# k = 192, l = 224, m = 256, n = 384
def reduction_A(x, input_channel, name): # 35x35 -> 17x17
    with tf.variable_scope(name) as scope:
        pool_1 = max_pool(x, [1,3,3,1], 2, 'VALID', 'reduction_A_pool_1') # 1    384

        conv_2 = conv_layer(x, [3,3,input_channel,384], [384], 2, 'VALID', 'reduction_A_conv_2') # 2 - 3x3   384

        conv_3_1 = conv_layer(x, [1,1,input_channel,192], [192], 1, 'SAME', 'reduction_A_conv_3_1') # 3 - 1x1
        conv_3_2 = conv_layer(conv_3_1, [3,3,192,224], [224], 1, 'SAME', 'reduction_A_conv_3_2') # 3 - 3x3
        conv_3_3 = conv_layer(conv_3_2, [3,3,224,256], [256], 2, 'VALID', 'reduction_A_conv_3_3') # 3 - 3x3     256

        return tf.concat([pool_1, conv_2, conv_3_3], axis=3, name='{}_concat'.format(name)) # 17x17x1024

def reduction_B(x, input_channel, name): # 17x17 -> 8x8
    with tf.variable_scope(name) as scope:
        pool_1 = max_pool(x, [1,3,3,1], 2, 'VALID', 'reduction_B_pool_1') # 1    1024

        conv_2_1 = conv_layer(x, [1,1,input_channel,192], [192], 1, 'SAME', 'reduction_B_conv_2_1') # 3 - 1x1
        conv_2_2 = conv_layer(conv_2_1, [3,3,192,192], [192], 2, 'VALID', 'reduction_B_conv_2_2') # 2 - 3x3   192

        conv_3_1 = conv_layer(x, [1,1,input_channel,256], [256], 1, 'SAME', 'reduction_B_conv_3_1') # 3 - 1x1
        conv_3_2 = conv_layer(conv_3_1, [1,7,256,256], [256], 1, 'SAME', 'reduction_B_conv_3_2') # 3 - 1x7
        conv_3_3 = conv_layer(conv_3_2, [7,1,256,320], [320], 1, 'SAME', 'reduction_B_conv_3_3') # 3 - 7x1
        conv_3_4 = conv_layer(conv_3_3, [3,3,320,320], [320], 2, 'VALID', 'reduction_B_conv_3_4') # 3 - 3x3     320

        return tf.concat([pool_1, conv_2_2, conv_3_4], axis=3, name='{}_concat'.format(name)) # 8x8x1536
