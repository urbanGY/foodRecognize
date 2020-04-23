import imgFetch
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# print('tensorflow version : ', tf.__version__)
# print(tf.test.is_gpu_available())

def showList(img_list, label_list):
    print("image path list length : ",len(img_list))
    print("label list length : ",len(label_list))
    print("test case 0")
    print(img_list[0])
    print(label_list[0])

def test():
    img_list, label_list = imgFetch.get_path_label("한과")
    showList(img_list, label_list)

test()
