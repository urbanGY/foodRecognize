import imgFetch
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print('tensorflow version : ', tf.__version__)
print('is gpu available??? pease.. : ',tf.test.is_gpu_available())

def showList(img_list, label_list):
    print("\n**************************")
    print("image path list length : ",len(img_list))
    print("label list length : ",len(label_list))
    print("test case 0")
    print(img_list[0])
    print(label_list[0])
    print(" ")

def _read_py_function(path, label):
    img = np.array(Image.open(path))
    label = np.array(label, dtype=np.float32)
    return img.astype(np.float32), label

def get_iterator_next_element():
    img_list, label_list = imgFetch.get_path_label("한과")

    batch_size = 1
    dataset = tf.data.Dataset.from_tensor_slices((img_list, label_list))
    dataset = dataset.map(lambda img_list, label_list: tuple(tf.py_func(_read_py_function,[img_list, label_list], [tf.float32, tf.float32])))

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(int(len(img_list) *0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    print("batch is ready")
    return iterator, next_element

# reuse를 사용하면 재사용이 된다. 그런데 여기에 그게 들어가는게 맞는지 모르겠음;
# graph로 variable보는법 알아서 나중에 어떻게 돌아가나 판단

def conv_layer(x, kernel_shape, bias_shape, stride, padding, name):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', kernel_shape, initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
        b = tf.get_variable('biases', bias_shape, initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
        return tf.nn.relu(conv + b)

def max_pool(x, kernel_shape, stride, padding, name):
    return tf.nn.max_pool(x, kernel_shape, strides=[1, stride, stride, 1], padding = padding)


def dcnn(x):
    # stem
    conv_1 = conv_layer(x, [3,3,3,32], [32], 2, 'VALID', 'conv_1')
    conv_2 = conv_layer(conv_1, [3,3,32,32], [32], 1, 'VALID', 'conv_2')
    conv_3 = conv_layer(conv_2, [3,3,32,64], [64], 1, 'SAME', 'conv_3')

    pool_4 = max_pool(conv_3, [1,3,3,1], 2, 'VALID', 'pool_4') # 1
    conv_4 = conv_layer(conv_3, [3,3,64,96], [96], 2, 'VALID', 'conv_4') # 2

    concat_5 = tf.concat([pool_4, conv_4], axis=3) # 73x73x160

    conv_5_1 = conv_layer(concat_5, [1,1,160,64], [64], 1, 'SAME', 'conv_5_1') # 1
    conv_6_1 = conv_layer(conv_5_1, [3,3,64,96], [96], 1, 'VALID', 'conv_6_1')

    conv_5_2 = conv_layer(concat_5, [1,1,160,64], [64], 1, 'SAME', 'conv_5_2') # 2
    conv_6_2 = conv_layer(conv_5_2, [7,1,64,64], [64], 1, 'SAME', 'conv_6_2')
    conv_7_2 = conv_layer(conv_6_2, [1,7,64,64], [64], 1, 'SAME', 'conv_7_2')
    conv_8_2 = conv_layer(conv_7_2, [3,3,64,96], [96], 1, 'VALID', 'conv_8_2')

    concat_9 = tf.concat([conv_6_1, conv_8_2], axis=3) # 71x71x192

    conv_10 = conv_layer(concat_9, [3,3,192,192], [192], 2, 'VALID', 'conv_10') # 1
    pool_10 = max_pool(concat_9, [1,2,2,1], 2, 'VALID', 'pool_10') # 1

    concat_11 = tf.concat([conv_10, pool_10], axis=3) # 71x71x192
    print("concat 11 shape : ",concat_11.shape)
    return concat_11

def test():
    iterator, next_element = get_iterator_next_element()

    img_width = 299
    img_height = 299
    img_channel = 3
    img_result = 150

    x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_channel])
    y = tf.placeholder(tf.float32, shape=[None, img_result])
    keep_prob = tf.placeholder(tf.float32)

    test = dcnn(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        for i in range(3):
            image, label = sess.run(next_element)
            _t = sess.run(test, feed_dict={x:image})

            # print("len : ",len(image))
            # print("image",image[0])
            # print("label",label[0])

if __name__ == "__main__":
    test()
