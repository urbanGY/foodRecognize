"""
학습 모듈
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import os
print('tensorflow version : ', tf.__version__)
print(tf.test.is_gpu_available())

class readyDataError(Exception):
    def __str__(self):
        return "*** data set is invalid ***\n"

def set_input():
    path_list = []
    label_list = []

    input_list = [['noodle','ramen','Img_050_', [1,0] ],['noodle','mak_noodle','Img_051_', [0,1]]]
    for input in input_list:
        category = input[0]
        kind = input[1]
        img_name = input[2]
        label = input[3]
        print("category : %s , kind : %s , img code : %s read start!"%(category,kind,img_name))
        for index in range(1000):
            if index < 10:
                num = '000'+str(index)
            elif index < 100:
                num = '00'+str(index)
            elif index < 1000:
                num = '0'+str(index)
            else :
                num = str(index)
            file_name = img_name + num + '.jpg'
            path = f'image/img/{category}/{kind}/{file_name}'
            path_list.append(path)
            label_list.append(label)
    return path_list, label_list

def read_path_list(path_list, label_list):
    img = []
    label = []
    for i in range(len(path_list)):
        path = path_list[i]
        if os.path.isfile(path):
            tmp = np.float32(cv2.imread(path, cv2.IMREAD_COLOR))
            img.append(tmp)
            label.append(label_list[i])
    return img, label

def cnn(x):
    x_image = x

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[10, 10, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    # h_conv1 = tf.nn.relu(tf.matmul(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # h_drop1 = tf.nn.dropout(h_pool1, keep_prob)


    W_conv2 = tf.Variable(tf.truncated_normal(shape=[10, 10, 64, 128], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]))
    # h_conv2 = tf.nn.relu(tf.matmul(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    W_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]))
    # h_conv3 = tf.nn.relu(tf.matmul(h_pool2, W_conv3) + b_conv3)
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    W_conv4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[512]))
    # h_conv4 = tf.nn.relu(tf.matmul(h_pool3, W_conv4) + b_conv4)
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)


    W_conv5 = tf.Variable(tf.truncated_normal(shape=[5, 5, 512, 512], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[512]))
    # h_conv5 = tf.nn.relu(tf.matmul(h_pool4, W_conv5) + b_conv5)
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[12 * 12 * 512, 768], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[768]))
    h_conv5_flat = tf.reshape(h_conv5, [-1, 12*12*512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[768, 384], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    W_fc3 = tf.Variable(tf.truncated_normal(shape=[384, 2], stddev=5e-2))
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[2]))
    logits = tf.matmul(h_fc2,W_fc3) + b_fc3

    y_pred = tf.nn.softmax(logits)

    return y_pred, logits



#******************************************************************#

path_list, tag_list = set_input()
img_list, label_list = read_path_list(path_list, tag_list)
print("img list len : %d , label list len : %s"%(len(img_list), len(label_list)))
if len(img_list) != len(label_list):
    raise readyDataError()
# img = np.stack(read_path_list(path_list))

img = np.asarray(img_list, dtype=np.float32)
label = np.asarray(label_list, dtype=np.float32)

# img = tf.convert_to_tensor(np_img, np.float32)
# img_list = img_list.astype('float32')


print("img and label np array setting ready!")

dataset = tf.data.Dataset.from_tensor_slices((img,label))
dataset = dataset.repeat()
dataset = dataset.shuffle(2500)
dataset = dataset.batch(32)

iterator = dataset.make_initializable_iterator()
# iterator = dataset.__iter__()
next_element = iterator.get_next()
print("batch is ready!")

image_width = 96
image_height = 96
image_channel = 3
image_result = 2 #이건 학습으로 들어간 음식 종류의 숫자

x = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel])
y = tf.placeholder(tf.float32, shape=[None, image_result])
keep_prob = tf.placeholder(tf.float32)

y_pred, logits = cnn(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    for i in range(5000):
        print('*', end='')
        _x, _y = sess.run(next_element)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:_x, y:_y, keep_prob:1.0})
            train_loss = sess.run(loss, feed_dict={x:_x, y:_y, keep_prob:1.0})
            print("\nstep %d  : train accuracy : %.4f, function loss : %.4f"%(i,train_accuracy,train_loss))
        sess.run(train_step, feed_dict={x:_x, y:_y, keep_prob: 0.9})
