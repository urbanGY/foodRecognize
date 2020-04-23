"""
서버에 붙는 실행 모듈
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import os
import json

def inception(x, input_channel, conv_1_out, conv_3_reduce_out, conv_3_out, conv_5_reduce_out, conv_5_out, pool_proj_out):
    with tf.variable_scope('inception') as scope:
        # 1x1 conv
        w_conv_1 = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_1_out], stddev=5e-2))
        b_conv_1 = tf.Variable(tf.constant(0.1, shape=[conv_1_out]))
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1)

        # 3x3 conv
        w_conv_3_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_3_reduce_out], stddev=5e-2))
        b_conv_3_reduce = tf.Variable(tf.constant(0.1, shape=[conv_3_reduce_out]))
        conv_3_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_3_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_reduce)

        w_conv_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, conv_3_reduce_out, conv_3_out], stddev=5e-2))
        b_conv_3 = tf.Variable(tf.constant(0.1, shape=[conv_3_out]))
        conv_3 = tf.nn.relu(tf.nn.conv2d(conv_3_reduce, w_conv_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3)

        # 3x3 conv
        w_conv_5_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_reduce = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_5_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_reduce)

        w_conv_5 = tf.Variable(tf.truncated_normal(shape=[5, 5, conv_5_reduce_out, conv_5_out], stddev=5e-2))
        b_conv_5 = tf.Variable(tf.constant(0.1, shape=[conv_5_out]))
        conv_5 = tf.nn.relu(tf.nn.conv2d(conv_5_reduce, w_conv_5, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5)

        # pooling
        pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        w_pool = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, pool_proj_out], stddev=5e-2))
        b_pool = tf.Variable(tf.constant(0.1, shape=[pool_proj_out]))
        pool_proj = tf.nn.relu(tf.nn.conv2d(pooling, w_pool, strides=[1, 1, 1, 1], padding='SAME') + b_pool)

        concat = tf.concat([conv_1, conv_3, conv_5, pool_proj], axis=3)
        return concat

def dcnn(x_image): # softmax 사용해서 loss 섞나?
    # name > data with/height input size -> output size // channel input size -> output size

    # conv 1    > 112 -> 56 // 3 -> 64
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[7, 7, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)
    print('h_conv1 shape : ',h_conv1.get_shape())

    # max pool 1   > 56 -> 28 // 64 -> 64
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('h_pool1 shape : ',h_pool1.get_shape())

    # conv 2    > 28 -> 28 // 64 -> 192
    W_conv2_1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 64], stddev=5e-2))
    b_conv2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2_1 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1)
    print('h_conv2_1 shape : ',h_conv2_1.get_shape())

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 192], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[192]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv2_1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    print('h_conv2 shape : ',h_conv2.get_shape())

    # max pool 2   > 28 -> 14 // 192 -> 192
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('h_pool2 shape : ',h_pool2.get_shape())

    #inception 1_a  > 14 -> 14 // 192 -> 256 (64 + 128 + 32 + 32)
    inception_1_a = inception(h_pool2, 192, 64, 96, 128, 16, 32, 32)
    print('inception_1_a shape : ',inception_1_a.get_shape())

    #inception 1_b  > 14 -> 14 // 256 -> 480 (128 + 192 + 96 + 64)
    inception_1_b = inception(inception_1_a, 256, 128, 128, 192, 32, 96, 64)
    print('inception_1_b shape : ',inception_1_b.get_shape())

    # max pool 3   > 14 -> 7 // 480 -> 480
    h_pool3 = tf.nn.max_pool(inception_1_b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('h_pool3 shape : ',h_pool3.get_shape())

    #inception 2_a  > 7 -> 7 // 480 -> 512 (192 + 208 + 48 + 64)
    inception_2_a = inception(h_pool3, 480, 192, 96, 208, 16, 48, 64)
    print('inception_2_a shape : ',inception_2_a.get_shape())

    #inception 2_b  > 7 -> 7 // 512 -> 512 (160 + 224 + 64 + 64)
    inception_2_b = inception(inception_2_a, 512, 160, 112, 224, 24, 64, 64)

    #inception 2_c  > 7 -> 7 // 512 -> 512 (128 + 256 + 64 + 64)
    inception_2_c = inception(inception_2_b, 512, 128, 128, 256, 24, 64, 64)

    #inception 2_d  > 7 -> 7 // 512 -> 528 (112 + 288 + 64 + 64)
    inception_2_d = inception(inception_2_c, 512, 112, 144, 288, 32, 64, 64)
    print('inception_2_d shape : ',inception_2_d.get_shape())

    #inception 2_e  > 7 -> 7 // 528 -> 832 (256 + 320 + 128 + 128)
    inception_2_e = inception(inception_2_d, 528, 256, 160, 320, 32, 128, 128)
    print('inception_2_e shape : ',inception_2_e.get_shape())

    aux_flatten = tf.reshape(inception_2_e, [-1, 7*7*832])
    aux_W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*832, 6], stddev=5e-2))
    aux_b_fc1 = tf.Variable(tf.constant(0.1, shape=[6]))

    auxiliary = tf.matmul(aux_flatten, aux_W_fc1) + aux_b_fc1

    # # max pool 4   > 14 -> 7 // 832 -> 832
    # h_pool4 = tf.nn.max_pool(inception_2_e, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #inception 3_a  > 7 -> 7 // 832 -> 832 (256 + 320 + 128 + 128)
    inception_3_a = inception(inception_2_e, 832, 256, 160, 320, 32, 128, 128)

    #inception 3_b  > 7 -> 7 // 832 -> 1024 (384 + 384 + 128 + 128)
    inception_3_b = inception(inception_3_a, 832, 384, 192, 384, 48, 128, 128)
    print('inception_3_b shape : ',inception_3_b.get_shape())

    # avg pool 5   > 7 -> 1 // 1024 -> 1024
    h_pool5 = tf.nn.avg_pool(inception_3_b, ksize = [1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
    print('h_pool5 shape : ',h_pool5.get_shape())

    # drop out 40% keep_prob:0.6
    dropout = tf.nn.dropout(h_pool5, keep_prob)
    flatten = tf.reshape(dropout, [-1, 1*1*1024])

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[1 * 1 * 1024, 6], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[6]))

    logits = tf.matmul(flatten, W_fc1) + b_fc1
    y_pred = tf.nn.softmax(logits)

    return y_pred

#******************************************************************#
def get_name(index):
    list = ['라면','자장면','짬뽕','된장찌개','김치찌개','순두부찌개']
    return list[index]

def print_prediction(y_prediction):
    list = ['라면','자장면','짬뽕','된장찌개','김치찌개','순두부찌개']
    for i in range(len(y_prediction)):
        print('%s : %.3f'%(list[i],y_prediction[i]*100))

def classification(image):
    image_width = 112
    image_height = 112
    image_channel = 3
    image_result = 6 #이건 학습으로 들어간 음식 종류의 숫자

    x = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel])
    y = tf.placeholder(tf.float32, shape=[None, image_result])
    keep_prob = tf.placeholder(tf.float32)

    input_image = np.float32(cv2.imread(image, cv2.IMREAD_COLOR))
    y_pred = dcnn(x)

    saver = tf.train.Saver()
    model_path = 'model/ver_1.ckpt'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        prediction = sess.run(y_pred, feed_dict={x: input_image, keep_prob: 1.0})
        index = sess.run(tf.argmax(y_pred, 1), feed_dict={x: input_image, keep_prob: 1.0})
        print_prediction(prediction)
        answer = get_name(index)
        return answer
