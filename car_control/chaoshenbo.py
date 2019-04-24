#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:08:58 2019

@author: pi
"""
import RPi.GPIO as GPIO
import time
from car_controler import FourWheelDriveCar




f4 = FourWheelDriveCar()

trig= 23
echo=24

GPIO.setmode(GPIO.BCM)
GPIO.setup(trig,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(echo,GPIO.IN)

time.sleep(1)

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
 
    
def distance():

        GPIO.output(trig,GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(trig,GPIO.LOW)
        while not GPIO.input(echo):
            pass
        t1=time.time()
        while GPIO.input(echo):
            pass
        t2=time.time()
        t3 = (t2-t1)*340/2
        print((t2-t1)*340/2)
        if t3 <= 0.3:
            f4.stop()
   

if __name__  == "__main__":
    
    derection =  input("Please input direction: ")

    if derection == "HL": 
            distance()
            time.sleep(1)


