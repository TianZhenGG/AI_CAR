# coding=utf-8

from __future__ import division
import time
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()

#每个舵机由两个阵脚的脉冲进行控制
mA1=8
mA2=9
mB1=10
mB2=11

pwm.set_pwm_freq(50)

class FourWheelDriveCar:
# Define the number of all the GPIO that will used for the 4wd car   
    def forward(self):
        pwm.set_pwm(mA1,0,1024)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,1024)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
    #向后，两个舵机向与forward（）向反的方向旋转
    def back(self):
        pwm.set_pwm(mA2,0,1024)
        pwm.set_pwm(mA1,0,0)
        pwm.set_pwm(mB2,0,1024)
        pwm.set_pwm(mB1,0,0)
        time.sleep(1)
    #原地左转，给右舵机向前速度，给左舵机向后速度
    def spin_left(self):
        pwm.set_pwm(mA1,0,0)
        pwm.set_pwm(mA2,0,1024)
        pwm.set_pwm(mB1,0,1024)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
    #原地右转，给左舵机向前速度，给右舵机向后速度
    def spin_right(self):
        pwm.set_pwm(mA1,0,1024)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,0)
        pwm.set_pwm(mB2,0,1024)
        time.sleep(1)
    #左转，通过频率差值改变转向角度
    def left(self):
        pwm.set_pwm(mA1,0,512)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,1024)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
    #右转，通过频率差值改变旋转角度
    def right(self):
        pwm.set_pwm(mA1,0,1024)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,512)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
    def stop(self):
        pwm.set_pwm(mA1,0,0)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,0)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
        
    def reset(self):
        pwm.set_pwm(mA1,0,0)
        pwm.set_pwm(mA2,0,0)
        pwm.set_pwm(mB1,0,0)
        pwm.set_pwm(mB2,0,0)
        time.sleep(1)
        
    def carMove(self, direction):
        '''
        Car move according to the input paramter - direction
        '''
        if direction == 'F':
            self.forward()
        elif direction == 'B':
            self.back()
        elif direction == 'L':
            self.left()
        elif direction == 'R':
            self.right()
        elif direction == 'BL':
            self.spin_left()
        elif direction == 'BR':
            self.spin_right()
        elif direction == 'S':
            self.stop()
        else:
            print("The input direction is wrong! You can just input: F, B, L, R, BL,BR or S")    

if __name__ == "__main__":
    raspcar = FourWheelDriveCar()
    while (True):
        direction = input("Please input direction: ")
        raspcar.carMove(direction)
      
