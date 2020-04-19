"""
이미지 크기를 학습에 적절한 크기로 조절하는 모듈

이미지 크기를 300*300 크기로 리 사이즈해 저장한다.
"""
#_*_coding: UTF-8_*_
import cv2
import numpy as np
import json
import os
print('import cv2 success. version : ',cv2.__version__)

def hangulFilePathImageRead ( filePath ) :
    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def hangulFilePathImageWrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False



def readImg(category, name, code):
    destPath = f'C:/Users/sfsfk/Desktop/develope/capstone2/image/resizeImg/{category}/{name}/'
    try:
        if not os.path.exists(destPath):
            os.makedirs(destPath)
    except OSError:
        print('Error: createing directory. '+ destPath)

    basePath = f'{category}/{name}/'
    fileName = f'{code}'

    for i in range(0,1001):
        if i < 10:
            index = "000"+str(i)
        elif i < 100:
            index = "00"+str(i)
        elif i < 1000:
            index = "0"+str(i)
        else:
            index = str(i)
        tmpFileName = fileName + index + '.jpg'
        path = basePath + tmpFileName
        try:
            img = hangulFilePathImageRead(path)
            resize_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        except:
            print("except in resize file : ",tmpFileName)
            continue

        destination = destPath + tmpFileName
        result = hangulFilePathImageWrite(destination, resize_img)
        if not result:
            print(category,name,tmpFileName," write is fail...")




if __name__=="__main__":
    f = open('fileStructure.json', mode='r', encoding='utf-8')
    json_site = json.load(f)
    inputList = json_site['inputList']
    f.close()
    for input in inputList:
        print("start",input["category"],input["name"])
        readImg(input["category"],input["name"],input["code"])
