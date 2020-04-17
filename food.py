"""
이미지에서 음식부분만 추출하는 모듈
이미지 크기를 112*112 크기로 resize한다.
gray 스케일로 변경 한뒤 이미지에서 원을 추출한다.
추출한 원중에 반지름이 가장 긴 원을 선택하고 그 원을 기준으로 정사각형 모양으로 이미지를 자른다.
자른 이미지를 다시 112*112 크기로 resize한뒤 리턴한다.
"""

import numpy as np
import cv2 as cv

def img_processing(shape_img, name):
    #이미지 불러온뒤 224*224로 resize
    shape_img = cv.resize(shape_img,(112,112),interpolation = cv.INTER_LINEAR_EXACT)

    #이미지를 gray 스케일로 변경 및 resize
    img_gray = cv.cvtColor(shape_img,cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray,(112,112),interpolation = cv.INTER_LINEAR_EXACT)

    img_gray = cv.medianBlur(img_gray,5)
    img_color = cv.cvtColor(img_gray,cv.COLOR_GRAY2BGR)

    #이미지에서 원을 찾는 함수
    circles = cv.HoughCircles(img_gray,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=35,minRadius=28,maxRadius=56)

    #원들의 정보를 리스트로 저장
    circles = np.uint16(np.around(circles))

    #원의 중심, 반지름 저장 변수
    y=0
    x=0
    radius = 0

    #반지름이 가장 큰 원을 찾는 반복문
    for c in circles[0,:]:

        if radius < c[2] :
            #원의 중심 좌표
            x = c[1]
            y = c[0]
            #원의 반지름
            radius = c[2]

    #추출된 원을 기준으로 정사각형으로 자름
    shape_img = shape_img[x-radius : x+radius , y-radius : y+radius]
    #자른 이미지를 112*112로 resize
    shape_img = cv.resize(shape_img,(112,112),interpolation = cv.INTER_LINEAR_EXACT)

    #자른 이미지 리턴
    return shape_img


shape_dir = 'C:/Users/jhkim/OneDrive/Desktop/img/noodle/ramen/Img_050_0'    #사진의 절대경로

#1~1000
for idx in range(0,1100): # range(사진 시작번호 , 사진 끝번호)
    if idx < 10 :
        file_name = '00'+str(idx)+'.jpg'
    elif idx < 100 :
        file_name = '0'+str(idx)+'.jpg'
    else:
        file_name = str(idx)+'.jpg'

#     file_name = str(idx)+'.jpg'

    shape_img = cv.imread(shape_dir+file_name)
#     cv.imshow("aa",shape_img)
    try:
         test(shape_img, file_name)
    except:
         print(idx)
#     test(shape_img, file_name)

    input_key = cv.waitKey(0)
    cv.destroyAllWindows()
