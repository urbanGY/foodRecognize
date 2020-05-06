import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from layers.conv_layer import conv_layer
from layers.avg_pool import avg_pool

# pool 의 size가 2 일지 3일지 모르겠다;
def inception_A(x, input_channel, name):
    with tf.variable_scope(name) as scope:
        pool_1 = avg_pool(x, [1,2,2,1], 1, 'SAME', 'inception_A_pool_1') # 1 - avg_pool
        conv_1_1 = conv_layer(pool_1, [1,1,input_channel,96], [96], 1, 'SAME', 'inception_A_conv_1_1') # 1 - 1x1

        conv_2_1 = conv_layer(x, [1,1,input_channel,96], [96], 1, 'SAME', 'inception_A_conv_2_1') # 2 - 1x1

        conv_3_1 = conv_layer(x, [1,1,input_channel,64], [64], 1, 'SAME', 'inception_A_conv_3_1') # 3 - 1x1
        conv_3_2 = conv_layer(conv_3_1, [3,3,64,96], [96], 1, 'SAME', 'inception_A_conv_3_2') # 3 - 3x3

        conv_4_1 = conv_layer(x, [1,1,input_channel,64], [64], 1, 'SAME', 'inception_A_conv_4_1') # 4 - 1x1
        conv_4_2 = conv_layer(conv_4_1, [3,3,64,96], [96], 1, 'SAME', 'inception_conv_A_4_2') # 4 - 3x3
        conv_4_3 = conv_layer(conv_4_2, [3,3,96,96], [96], 1, 'SAME', 'inception_conv_A_4_3') # 4 - 3x3

        return tf.concat([conv_1_1, conv_2_1, conv_3_2, conv_4_3], axis=3, name='{}_concat'.format(name)) # 35x35x384

def inception_B(x, input_channel, name):
    with tf.variable_scope(name) as scope:
        pool_1 = avg_pool(x, [1,2,2,1], 1, 'SAME', 'inception_B_pool_1') # 1 - avg_pool
        conv_1_1 = conv_layer(pool_1, [1,1,input_channel,128], [128], 1, 'SAME', 'inception_B_conv_1_1') # 1 - 1x1

        conv_2_1 = conv_layer(x, [1,1,input_channel,96], [96], 1, 'SAME', 'inception_conv_2_1') # 2 - 1x1

        conv_3_1 = conv_layer(x, [1,1,input_channel,64], [64], 1, 'SAME', 'inception_conv_3_1') # 3 - 1x1
        conv_3_2 = conv_layer(conv_3_1, [3,3,64,96], [96], 1, 'SAME', 'inception_conv_3_2') # 3 - 3x3

        conv_4_1 = conv_layer(x, [1,1,input_channel,64], [64], 1, 'SAME', 'inception_conv_4_1') # 4 - 1x1
        conv_4_2 = conv_layer(conv_4_1, [3,3,64,96], [96], 1, 'SAME', 'inception_conv_4_2') # 4 - 3x3
        conv_4_3 = conv_layer(conv_4_2, [3,3,96,96], [96], 1, 'SAME', 'inception_conv_4_3') # 4 - 3x3

        return tf.concat([conv_1_1, conv_2_1, conv_3_2, conv_4_3], axis=3, name='{}_concat'.format(name)) # 35x35x384
