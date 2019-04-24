# coding = utf-8
import cv2
import socket
import face_recognition1
import dlib
import time

addr = ('192.168.43.236',7070)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


while True:
    # 创建视频对象，打开摄像头
    video_capture = cv2.VideoCapture(1)
    ret, frame = video_capture.read()
    # 释放视频对象
    video_capture.release()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 这里将分辨率缩小为1/4，故比例系数增大为4倍，现在是0.078125*4 = 0.3125
    output = frame[:, :, ::-1]
    face_locations = face_recognition1.face_locations(output)
    if face_locations:
        x = (face_locations[0][1] + face_locations[0][3]) / 2
        y = (face_locations[0][0] + face_locations[0][2]) / 2
        print(x, y)  # 输出脸中心到右上顶点的水平和垂直距离
    else:
        x, y = 55, 42  # 如果没有脸则让舵机保持不动，相当于脸在中央（这时的分辨率为160*120）

    data = str(x) + ',' + str(y)

    s.sendto(data.encode('utf-8'), addr)
