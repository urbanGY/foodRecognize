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

        conv_2_1 = conv_layer(x, [1,1,input_channel,384], [384], 1, 'SAME', 'inception_B_conv_2_1') # 2 - 1x1

        conv_3_1 = conv_layer(x, [1,1,input_channel,192], [192], 1, 'SAME', 'inception_B_conv_3_1') # 3 - 1x1
        conv_3_2 = conv_layer(conv_3_1, [1,7,192,224], [224], 1, 'SAME', 'inception_B_conv_3_2') # 3 - 1x7
        conv_3_3 = conv_layer(conv_3_2, [7,1,224,256], [256], 1, 'SAME', 'inception_B_conv_3_3') # 3 - 7x1

        conv_4_1 = conv_layer(x, [1,1,input_channel,192], [192], 1, 'SAME', 'inception_B_conv_4_1') # 4 - 1x1
        conv_4_2 = conv_layer(conv_4_1, [1,7,192,192], [192], 1, 'SAME', 'inception_B_conv_4_2') # 4 - 1x7
        conv_4_3 = conv_layer(conv_4_2, [7,1,192,224], [224], 1, 'SAME', 'inception_B_conv_4_3') # 4 - 7x1
        conv_4_4 = conv_layer(conv_4_3, [1,7,224,224], [224], 1, 'SAME', 'inception_B_conv_4_4') # 4 - 1x7
        conv_4_5 = conv_layer(conv_4_4, [1,7,224,256], [256], 1, 'SAME', 'inception_B_conv_4_5') # 4 - 7x1

        return tf.concat([conv_1_1, conv_2_1, conv_3_3, conv_4_5], axis=3, name='{}_concat'.format(name)) # 17x17x1024

def inception_C(x, input_channel, name):
    with tf.variable_scope(name) as scope:
        pool_1 = avg_pool(x, [1,2,2,1], 1, 'SAME', 'inception_C_pool_1') # 1 - avg_pool
        conv_1_1 = conv_layer(pool_1, [1,1,input_channel,256], [256], 1, 'SAME', 'inception_C_conv_1_1') # 1 - 1x1

        conv_2_1 = conv_layer(x, [1,1,input_channel,256], [256], 1, 'SAME', 'inception_C_conv_2_1') # 2 - 1x1

        conv_3_1 = conv_layer(x, [1,1,input_channel,384], [384], 1, 'SAME', 'inception_C_conv_3_1') # 3 - 1x1
        conv_3_2_1 = conv_layer(conv_3_1, [1,3,384,256], [256], 1, 'SAME', 'inception_C_conv_3_2_1') # 3 - 1x3
        conv_3_2_2 = conv_layer(conv_3_1, [3,1,384,256], [256], 1, 'SAME', 'inception_C_conv_3_2_2') # 3 - 3x1

        conv_4_1 = conv_layer(x, [1,1,input_channel,384], [384], 1, 'SAME', 'inception_C_conv_4_1') # 4 - 1x1
        conv_4_2 = conv_layer(conv_4_1, [1,3,384,448], [448], 1, 'SAME', 'inception_C_conv_4_2') # 4 - 1x3
        conv_4_3 = conv_layer(conv_4_2, [3,1,448,512], [512], 1, 'SAME', 'inception_C_conv_4_3') # 4 - 3x1
        conv_4_4_1 = conv_layer(conv_4_3, [3,1,512,256], [256], 1, 'SAME', 'inception_C_conv_4_4_1') # 4 - 3x1
        conv_4_4_2 = conv_layer(conv_4_3, [1,3,512,256], [256], 1, 'SAME', 'inception_C_conv_4_4_2') # 4 - 1x3

        return tf.concat([conv_1_1, conv_2_1, conv_3_2_1, conv_3_2_2, conv_4_4_1, conv_4_4_2], axis=3, name='{}_concat'.format(name)) # 8x8x1536
