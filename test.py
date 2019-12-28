import json


def make_onehot(length, index):
    tmp = []
    for i in range(length):
        tmp.append(0)
    tmp[index] = 1
    return tmp


list =  []
with open('input_data/input.json') as file:
    json_data = json.load(file)
    input_list = json_data['input_list']
    input_len = len(input_list)
    index = 0
    for input in input_list:
        category = input['category']
        kind = input['kind']
        img_name = input['img_name']
        label = make_onehot(input_len, index)
        index += 1
        list.append([category,kind,img_name,label])

for l in list:
    print(l)
