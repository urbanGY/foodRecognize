import glob
import json
import numpy as np
from PIL import Image

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
    onehot = onehot.astype(np.uint8)
    return onehot

def read_image(path):
    img = np.array(Image.open(path))
    return img

def _read_py_function(path, label):
    img = read_image(path)
    label = np.array(label, dtype=np.uint8)
    return img.astype(np.float32), label

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

# batch_size = 16
# dataset = tf.data.Dataset.from_tensor_slice((img_list, label_list))
# dataset = dataset.map(lambda img_list, label_list: tuple(tf.py_func(_read_py_function,[img_list, label_list], [tf.float32, tf.uint8])))
#
# # dataset = dataset.repeat()
# dataset = dataset.shuffle(buffer_size=(int(len(data_list) *0.4) + 3 * batch_size))
# dataset = dataset.batch(batch_size)
#
# iterator = dataset.make_initializable_iterator()
# image_stacked, label_stacked = iterator.get_next()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     image, label = sess.run([image_stacked, label_stacked])
#     print("done")
