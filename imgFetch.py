import glob
import json
import numpy as np

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
