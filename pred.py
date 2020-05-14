from imgFetch import get_iterator_next_element
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from layers.conv_layer import conv_layer
from layers.max_pool import max_pool
from layers.avg_pool import avg_pool
from layers.fc_layer import fc_layer

from layers.inception import inception_A, inception_B, inception_C
from layers.reduction import reduction_A, reduction_B


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print('tensorflow version : ', tf.__version__)
print('is gpu available??? : ',tf.test.is_gpu_available())

def dcnn(x, keep_prob, img_result):
    # stem
    conv_1 = conv_layer(x, [3,3,3,32], [32], 2, 'VALID', 'conv_1')
    conv_2 = conv_layer(conv_1, [3,3,32,32], [32], 1, 'VALID', 'conv_2')
    conv_3 = conv_layer(conv_2, [3,3,32,64], [64], 1, 'SAME', 'conv_3')

    pool_4 = max_pool(conv_3, [1,3,3,1], 2, 'VALID', 'pool_4') # 1
    conv_4 = conv_layer(conv_3, [3,3,64,96], [96], 2, 'VALID', 'conv_4') # 2

    concat_5 = tf.concat([pool_4, conv_4], axis=3, name='concat_5') # 73x73x160

    conv_5_1 = conv_layer(concat_5, [1,1,160,64], [64], 1, 'SAME', 'conv_5_1') # 1
    conv_6_1 = conv_layer(conv_5_1, [3,3,64,96], [96], 1, 'VALID', 'conv_6_1')

    conv_5_2 = conv_layer(concat_5, [1,1,160,64], [64], 1, 'SAME', 'conv_5_2') # 2
    conv_6_2 = conv_layer(conv_5_2, [7,1,64,64], [64], 1, 'SAME', 'conv_6_2')
    conv_7_2 = conv_layer(conv_6_2, [1,7,64,64], [64], 1, 'SAME', 'conv_7_2')
    conv_8_2 = conv_layer(conv_7_2, [3,3,64,96], [96], 1, 'VALID', 'conv_8_2')

    concat_9 = tf.concat([conv_6_1, conv_8_2], axis=3, name='concat_9') # 71x71x192

    conv_10 = conv_layer(concat_9, [3,3,192,192], [192], 2, 'VALID', 'conv_10') # 1
    pool_10 = max_pool(concat_9, [1,2,2,1], 2, 'VALID', 'pool_10') # 1

    concat_11 = tf.concat([conv_10, pool_10], axis=3, name='concat_11') # 71x71x192
    # print("stem shape : ",concat_11.shape)

    inception_a_12 = inception_A(concat_11, 384, 'inception_a_12')
    inception_a_13 = inception_A(inception_a_12, 384, 'inception_a_13')
    inception_a_14 = inception_A(inception_a_13, 384, 'inception_a_14')
    inception_a_15 = inception_A(inception_a_14, 384, 'inception_a_15') # 4번 반복
    # print("inception A shape : ",inception_a_15.shape)

    reduction_a_16 = reduction_A(inception_a_15, 384, 'reduction_a_16')
    # print("reduction A shape : ",reduction_a_16.shape)

    inception_b_17 = inception_B(reduction_a_16, 1024, 'inception_b_17')
    inception_b_18 = inception_B(inception_b_17, 1024, 'inception_b_18')
    inception_b_19 = inception_B(inception_b_18, 1024, 'inception_b_19')
    inception_b_20 = inception_B(inception_b_19, 1024, 'inception_b_20')
    inception_b_21 = inception_B(inception_b_20, 1024, 'inception_b_21')
    inception_b_22 = inception_B(inception_b_21, 1024, 'inception_b_22')
    inception_b_23 = inception_B(inception_b_22, 1024, 'inception_b_23')
    # print("inception B shape : ",inception_b_23.shape)

    reduction_b_24 = reduction_B(inception_b_23, 1024, 'reduction_b_24')
    # print("reduction B shape : ",reduction_b_24.shape)

    inception_c_25 = inception_C(reduction_b_24, 1536, 'inception_c_25')
    inception_c_26 = inception_C(inception_c_25, 1536, 'inception_c_26')
    inception_c_27 = inception_C(inception_c_26, 1536, 'inception_c_27')
    # print("inception C shape : ",inception_c_27.shape)

    pool_28 = avg_pool(inception_c_27, [1,8,8,1], 1, 'VALID', 'avg_pool_28') # 1 - avg_pool

    dropout = tf.nn.dropout(pool_28, keep_prob)
    flatten = tf.reshape(dropout, [-1, 1*1*1536])

    logits = fc_layer(flatten, 1536, img_result,'fc_layer')
    print("logits : ",logits.shape)

    return tf.nn.softmax(logits)

def get_name(index):
    list = [] #이름 후보군 원 핫 인코딩 복호화?... 하는거 ㅋㅋ
    return list[index]

def print_prediction(y_pred):
    list = [] #이름 후보군
    for i in range(len(y_prediction)):
        print('%s : %.3f'%(list[i],y_prediction[i]*100))

def prediction():
    #input_image 만들기 img_fetch 에서 이미지 랜덤으로 가져오는거
    img_width = 299
    img_height = 299
    img_channel = 3
    img_result = 150

    x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_channel])
    y = tf.placeholder(tf.float32, shape=[None, img_result])
    keep_prob = tf.placeholder(tf.float32)

    y_pred = dcnn(x, keep_prob, img_result)

    saver = tf.train.Saver()
    model_path = 'export_model/'
    index = '0'
    file_name = '/g_4.ckpt'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path+index+file_name)
        prediction = sess.run(y_pred, feed_dict={x: input_image, keep_prob: 1.0})
        index = sess.run(tf.argmax(y_pred, 1), feed_dict={x: input_image, keep_prob: 1.0})
        print_prediction(prediction)
        answer = get_name(index)
        return answer

if __name__ == "__main__":
    test()
    print("finish!")
