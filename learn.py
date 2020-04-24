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

    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices((img_list, label_list))
    dataset = dataset.map(lambda img_list, label_list: tuple(tf.py_func(_read_py_function,[img_list, label_list], [tf.float32, tf.float32])))

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(int(len(img_list) *0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    print("batch is ready")
    return iterator, next_element

def test():
    iterator, next_element = get_iterator_next_element()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(3):
            image, label = sess.run(next_element)
            print("len : ",len(image))
            print("image",image[0])
            print("label",label[0])

if __name__ == "__main__":
    test()
