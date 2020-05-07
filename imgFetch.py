import glob
import json
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def make_list():
    f = open('image/fileStructure.json', mode='r', encoding='utf-8')
    json_list = json.load(f)
    inputList = json_list['inputList']
    name_list = []
    for input in inputList:
        name = input["name"]
        name_list.append(name)
    return np.unique(name_list)

def get_name(path):
    name = path.split('\\')[-2]
    return name

def make_onehot(name_list, path):
    onehot = name_list == get_name(path)
    onehot = onehot.astype(np.float32)
    return onehot

def get_path_label(category):
    # controll image scope argument category
    if category == "all":
        img_list = glob.glob('image\\resizeImg\\*\\*\\*.jpg', recursive=True)
    else :
        img_list = glob.glob(f'image\\resizeImg\\{category}\\*\\*.jpg', recursive=True)

    # name list for make one hot encoding
    name_list = make_list()

    # make one hot encoding label list
    label_list = [make_onehot(name_list, path).tolist() for path in img_list]

    return img_list, label_list

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


def get_iterator_next_element(category, batch_size):
    img_list, label_list = get_path_label(category)

    dataset = tf.data.Dataset.from_tensor_slices((img_list, label_list))
    dataset = dataset.map(lambda img_list, label_list: tuple(tf.py_func(_read_py_function,[img_list, label_list], [tf.float32, tf.float32])))

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(int(len(img_list) *0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    print("batch is ready")
    return iterator, next_element
