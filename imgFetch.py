import glob
import json
import numpy as np
from PIL import Image
import tensorflow as tf

img_list = glob.glob('image\\resizeImg\\**\\*.jpg', recursive=True)

def make_list():
    f = open('image/fileStructure.json', mode='r', encoding='utf-8')
    json_list = json.load(f)
    inputList = json_list['inputList']
    name_list = []
    for input in inputList:
        name = input["name"]
        name_list.append(name)
    return np.unique(name_list)

name_list = make_list()

def get_name(path):
    name = path.split('\\')[-2]
    return name

def make_onehot(path):
    onehot = name_list == get_name(path)
    onehot = onehot.astype(np.uint8)
    return onehot

def read_image(path):
    img = np.array(Image.open(path))
    return img

def _read_py_function(path, label):
    img = read_image(path)
    label = np.array(label, dtype=np.uint8)
    return img.astype(np.float32), label

batch_size = 16
label_list = [make_onehot(path).tolist() for path in img_list]

dataset = tf.data.Dataset.from_tensor_slice((img_list, label_list))
dataset = dataset.map(lambda img_list, label_list: tuple(tf.py_func(_read_py_function,[img_list, label_list], [tf.float32, tf.uint8])))

# dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=(int(len(data_list) *0.4) + 3 * batch_size))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
image_stacked, label_stacked = iterator.get_next()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    image, label = sess.run([image_stacked, label_stacked])
    print("done")
