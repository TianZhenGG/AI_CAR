
# coding = utf-8

import RPi.GPIO as GPIO
import time
import socket



trig= 23
echo=24
 
def setServoAngle(servo, angle):
    '''
    :param servo 控制舵机的引脚编号，这取决于你，我用的分别是17和27
    :param angle 舵机的位置，范围是：0到180，单位是度
    return: None
    '''
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(servo, GPIO.OUT)
    pwm = GPIO.PWM(servo, 50)
    pwm.start(8)
    dutyCycle = angle / 18. + 3.
    pwm.ChangeDutyCycle(dutyCycle)
    time.sleep(0.3)
    pwm.stop()
    GPIO.cleanup()
 
    
# 设置舵机的初始位置
x0, y0 = 55, 42
setServoAngle(27, x0)
setServoAngle(17, y0)

addr = ('192.168.43.76',7070)
addr2 = ('192.168.43.76',6060)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind(addr)
s2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s2.bind(addr2)
s2.listen(2)
s2,addr2 = s2.accept()

while 1:
       
    data,addr = s.recvfrom(2048)
    x = data.decode()
    print(x)
    strX = str(x)
    arr = strX.split(',')
    intX = int(float(arr[0]))
    intY = int(float(arr[1]))

    x = intX
    y = intY
          
    dx = (56 - x) * 0.37
    dy = -(42 - y) * 0.37
 
    if abs(dx) >= 6:  # 设置一个阈值，当角度大于3时，才移动，避免舵机一直在原地抖动，下同
        x0 += dx
        if x0 > 120:  # 设置界限，超出范围不再转动，下同
            x0 = 120
        elif x0 < 0:
            x0 = 0
        setServoAngle(27, x0) # 水平方向的舵机控制
        
       
    
    if abs(dy) >= 6:  # 设置阈值
        y0 += dy
        if y0 > 50:
            y0 = 50
        elif y0 < 0:
            y0 = 0
        setServoAngle(17, y0)  # 垂直方向的舵机控制
               
        
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trig,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(echo,GPIO.IN)
    GPIO.output(trig,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(trig,GPIO.LOW)
    GPIO.output(trig,GPIO.HIGH)

    GPIO.output(trig,GPIO.LOW)
    while not GPIO.input(echo):
        pass
    t1=time.time()
    while GPIO.input(echo):
        pass
    t2=time.time()
    t3 = (t2-t1)*340/2    
    data = str(t3)
    s2.send(data.encode())
    time.sleep(1)
    setServoAngle(27, 55)
    setServoAngle(17, 42)

  
s2.close()        
    






