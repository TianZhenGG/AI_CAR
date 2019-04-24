import os
import sys
import cv2
import socket
import face_recognition
import dlib
import pandas as np


client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = "192.168.43.76"
port = 7070
client.connect((host,port))


while True:
    video_capture = cv2.VideoCapture(1)
    ret, frame = video_capture.read()
    # 释放视频对象
    video_capture.release()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    output = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(output)
    if face_locations:
        x = (face_locations[0][1] + face_locations[0][3]) / 2
        y = (face_locations[0][0] + face_locations[0][2]) / 2
        # 输出脸中心到右上顶点的水平和垂直距离
        # driver.find_element_by_xpath("./*//button[@id='F']").click()
    else:
        x, y = 55, 42  # 如果没有脸则让舵机保持不动，相当于脸在中央（这时的分辨率为160*120）
        break

    data = str(x) + ',' + str(y)
    client.send(data.encode('utf-8'))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
client.close()
cv2.destroyAllWindows()
