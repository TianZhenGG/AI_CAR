import os
import sys
import cv2
import socket
import face_recognition
import dlib
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import matplotlib.pyplot as plt

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


def update():
    for episode in range(100):
        # 初始化 state（状态）
        state = env.reset()

        step_count = 0  # 记录走过的步数

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 挑选 action
            action = RL.choose_action(str(state))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state, reward 和 done (是否是踩到炸弹或者找到宝藏)
            state_, reward, done = env.step(action)

            step_count += 1  # 增加步数

            # 机器人大脑从这个过渡（transition） (state, action, reward, state_) 中学习
            RL.learn(str(state), action, reward, str(state_))

            # 机器人移动到下一个 state
            state = state_

            # 如果踩到炸弹或者找到宝藏, 这回合就结束了
            if done:
                print("回合 {} 结束. 总步数 : {}\n".format(episode + 1, step_count))
                break
    # 结束游戏并关闭窗口
    print('游戏结束')
    env.destroy()
    print('\nQ 表:')
    print(RL.q_table)


def playdate():
    for episode in range(1):
        # 初始化 state（状态）
        state = env1.reset()

        step_count = 0  # 记录走过的步数

        while True:
            # 更新可视化环境
            env1.render()

            # RL 大脑根据 state 挑选 action
            action = RL.choose_action(str(state))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state, reward 和 done (是否是踩到炸弹或者找到宝藏)
            state_, reward, done = env1.step(action)

            step_count += 1  # 增加步数

            # 机器人移动到下一个 state
            state = state_

            # 如果踩到炸弹或者找到宝藏, 这回合就结束了
            if done:
                print("回合 {} 结束. 总步数 : {}\n".format(episode + 1, step_count))
                break

    env1.destroy()


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

driver = webdriver.Chrome(executable_path='/home/tianzhen/Downloads/chromedriver')
driver.get('http://192.168.43.76:9090/')

addr = ("192.168.43.76",7070)
addr2 = ("192.168.43.76",6060)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s2.connect(addr2)
#print(x)

while True:
    image_1 = cv2.VideoCapture('http://192.168.43.76:8080/?action=stream')
    ret1, image1 = image_1.read()
    image_1.release()
    img1 = image1[:, :, ::-1]
    results1 = model.detect([img1], verbose=0)
    r1 = results1[0]

    # Show the frame of video on the screen
    #cv2.imshow('Video', )
    #visualize.display_instances(img1, r['rois'], r['masks'], r['class_ids'],
    #                   class_names, r['scores'])
    if r1['class_ids'] is not None:
        print("============")
        print("有东西")
        print("============")
        list1 = []
        for i in r1['class_ids']:
            print('发现了', class_names[i])
            print(r1['rois'])
            if i == 1:
                #driver.find_element_by_xpath("./*//button[@id='HL']").click()
                while True:
                    video_capture = cv2.VideoCapture('http://192.168.43.76:8080/?action=stream')
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
                    data3 = str(x) + ',' + str(y)
                    s.sendto(data3.encode(),addr)
                    #f_btn = driver.find_element_by_xpath('./*//button[@id="F"]')
                    #ActionChains(driver).click_and_hold(f_btn).perform()
                    #driver.find_element_by_xpath("./*//button[@id='HL']").click()

        for k in range(len(r1['class_ids'])):

            x1 = (r1['rois'][k][3] - r1['rois'][k][1])/2 + (r1['rois'][k][1])
            y1 = (r1['rois'][k][2] - r1['rois'][k][0])/2 + (r1['rois'][k][0])

            #visualize.display_instances(img1, r1['rois'], r1['masks'], r1['class_ids'],
                                       # class_names, r1['scores'])

            # y0 = (r1['rois'][2]-r1['rois'][0])/2
            # z = (x0,y0)
            # list1.append(class_names[i])
            x0 = float(x1)*0.25
            y0 = float(y1)*0.25
            print(x0 , y0)
            data = str(x0) + ',' + str(y0)
            s.sendto(data.encode(), addr)
            #driver.find_element_by_xpath("./*//button[@id='HL']").click()
            time.sleep(2)
            chaodata = s2.recv(1024)
            chaoshendata = chaodata.decode()

            x3 = str(chaoshendata)
            x2 = (int(r1['rois'][k][3]) - int(r1['rois'][k][1]))
            list1.append([(x2,x0,y0,x3)])
            datas = np.array(list1)
            np.save('demo.npy',datas)

            #visualize.display_instances(img1, r1['rois'], r1['masks'], r1['class_ids'],
                            # class_names, r1['scores'])

        sys.path.append(os.path.join(ROOT_DIR, "Q_learning_Maze"))
        from Q_learning_Maze.q_learning import QLearning
        from Q_learning_Maze import play
        from Q_learning_Maze.env import Maze
        from Q_learning_Maze.env1 import Maze1

        env = Maze()
        RL = QLearning(actions=list(range(env.n_actions)))
        # 开始可视化环境
        env.after(100, update)
        env.mainloop()

        env1 = Maze1()
        print('开始走啦')

        # 开始可视化环境
        env1.after(1, playdate)
        env1.mainloop()


cv2.destroyAllWindows()






