import os
import sys
import cv2
import socket
import face_recognition
import dlib
import pandas as np
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import matplotlib.pyplot as plt


#addr = ('192.168.43.236',7070)
#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import mrcnn
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import MaskRCNN
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['背景', '人', 'bicycle', '车', 'motorcycle', '飞机',
               '大巴车', '火车', 'truck', '轮船', '红绿灯',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               '猫', '狗', '马', '绵羊', 'cow', '大象', 'bear',
               'zebra', '长颈鹿', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', '瓶子', 'wine glass', 'cup',
               '叉子', '小刀', 'spoon', '碗', '香蕉', '苹果',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', '蛋糕', '椅子', 'couch', 'potted plant', 'bed',
               'dining table', '厕所', '电视', 'laptop', '鼠标', 'remote',
               '键盘', '手机', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', '书', '时钟', 'vase', 'scissors',
               'teddy bear', 'hair drier', '牙刷']


knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 12))

#driver = webdriver.Chrome(executable_path='/home/tianzhen/Downloads/chromedriver')
#driver.get('http://192.168.43.236:9090/')

#data,address = s.recvfrom(2048)
#x = data.decode()
#print(x)
while True:
    image_1 = cv2.VideoCapture(0)
    image_2 = cv2.VideoCapture(1)

    ret1, image1 = image_1.read()
    ret2, image2 = image_2.read()

    image_1.release()
    image_2.release()

    img1 = image1[:, :, ::-1]
    img2 = image2[:, :, ::-1]
    results1 = model.detect([img1], verbose=0)
    results2 = model.detect([img2], verbose=0)
    r1 = results1[0]
    r2 = results2[0]

    # Show the frame of video on the screen
    #cv2.imshow('Video', )
    #visualize.display_instances(img1, r['rois'], r['masks'], r['class_ids'],
    #                   class_names, r['scores'])
    if r1['class_ids'] is not None:

        print("============")

        if list(set(r1['class_ids']).intersection(set(r2['class_ids']))) :
            print("============")
            print("有东西")
            print("============")

            for i in r1['class_ids']:
                list1 = []
                if i == 1:
                    #driver.find_element_by_xpath("./*//button[@id='HL']").click()
                    while True:
                        video_capture = cv2.VideoCapture(0)
                        ret, frame = video_capture.read()
                        # 释放视频对象
                        video_capture.release()
                        frame = cv2.resize(frame, (0, 0), fx=0.25,fy=0.25)
                        output = frame[:, :, ::-1]
                        face_locations = face_recognition.face_locations(output)
                        if face_locations:
                            x = (face_locations[0][1] + face_locations[0][3]) / 2
                            y = (face_locations[0][0] + face_locations[0][2]) / 2
                            # 输出脸中心到右上顶点的水平和垂直距离
                            #driver.find_element_by_xpath("./*//button[@id='F']").click()
                        else:
                            x, y = 55, 42  # 如果没有脸则让舵机保持不动，相当于脸在中央（这时的分辨率为160*120）
                            break
                        #data = str(x) + ',' + str(y)
                        #s.sendto(data.encode('utf-8'), addr)
                        #f_btn = driver.find_element_by_xpath('./*//button[@id="F"]')
                        #ActionChains(driver).click_and_hold(f_btn).perform()
                        #driver.find_element_by_xpath("./*//button[@id='HL']").click()


                else:

                    print('发现了',class_names[i])

                    #y0 = (r1['rois'][2]-r1['rois'][0])/2
                    #z = (x0,y0)
                    #list1.append(class_names[i])
                    y0 = r1['rois'][0][0]
                    x0 = r1['rois'][0][1]

                    y1 = r1['rois'][0][2]
                    x1 = r1['rois'][0][3]

                    x = (((int(x1) -int(x0))/2) + x0)/10
                    y = (((int(y1) -int(y0))/2) + y0)/10

                    list1.append((class_names[i],(x,y)))
                    print("============")
                    print(list1)
                    #data = str(x) + ',' + str(y)
                    #s.sendto(data.encode('utf-8'), addr)
                    #visualize.display_instances(img1, r1['rois'], r1['masks'], r1['class_ids'],
                     #                 class_names, r1['scores'])



    else:
            print('看错了')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
    
cv2.destroyAllWindows()




