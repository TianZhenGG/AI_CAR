# -*- coding: UTF-8 -*-

"""
Q Learning 例子的 Maze（迷宫） 环境

黄色圆形 :   机器人
红色方形 :   炸弹     [reward = -1]
绿色方形 :   宝藏     [reward = +1]
其他方格 :   平地     [reward = 0]
"""
import os
import sys
import time
import numpy as np
import pickle
import pathlib

# Python2 和 Python3 中 Tkinter 的名称不一样
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

WIDTH = 7  # 迷宫的宽度
HEIGHT = 9  # 迷宫的高度
UNIT = 40  # 每个方块的大小（像素值）

a = np.load('/home/tianzhen/PycharmProjects/AI_CAR/samples/demo.npy')
b = a.tolist()

list1 = []
list2 = []
list3 = []

for i in b:

    for j in i:
        if float(j[3]) < 5:
            dx = (55 - (float((j[1])))) * 0.3
            d = int((0.0029 * float(j[0])) / (float(j[3]) * np.cos(float(dx)))) + 1

        else:
            dx = (55 - (float((j[1])))) * 0.3
            d = int((0.0029 * float(j[0])) / (4 * np.cos(float(dx)))) + 1
        list1.append(d)
        x0 = int(0.6 * (7 - float(j[3]) * np.cos(float(dx))))

        list2.append(x0)
        y0 = int((2 / 3) * (9 - float(j[3]) * np.sin(float(dx)))) - 2

        list3.append(y0)


# 迷宫
class Maze1(tk.Tk, object):
    def __init__(self):
        super(Maze1, self).__init__()
        self.action_space = ['u', 'l', 'r']  # 上，左，右 四个 action（动作）
        self.n_actions = len(self.action_space)  # action 的数目
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))  # Tkinter 的几何形状
        self.build_maze()
        for k in range(len(list2)):
            if k + 1 == 1:
                for s in range(list1[0]):
                    if s + 1 == 1:
                        self._draw_rect1(list2[0] + s, list3[0], 'black')

                    elif s + 1 == 2:
                        self._draw_rect11(list2[0] + s, list3[0], 'black')

                    elif s + 1 == 3:
                        self._draw_rect12(list2[0] + s, list3[0], 'black')
            elif k + 1 == 2:
                for s in range(list1[1]):
                    if s + 1 == 1:
                        self._draw_rect2(list2[1] + s, list3[1], 'black')

                    elif s + 1 == 2:
                        self._draw_rect13(list2[1] + s, list3[1], 'black')

                    elif s + 1 == 3:
                        self._draw_rect14(list2[1] + s, list3[1], 'black')
            elif k + 1 == 3:
                for s in range(list1[2]):
                    if s + 1 == 1:
                        self._draw_rect3(list2[2] + s, list3[2], 'black')

                    elif s + 1 == 2:
                        self._draw_rect15(list2[2] + s, list3[2], 'black')

                    elif s + 1 == 3:
                        self._draw_rect16(list2[2] + s, list3[2], 'black')

    def _draw_rect1(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect1 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect1

    def _draw_rect2(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect2 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect2

    def _draw_rect3(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect3 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect3

    def _draw_rect4(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect4 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect4

    def _draw_rect5(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect5 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect5

    def _draw_rect10(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect10 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect10

    def _draw_rect11(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect11 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect11

    def _draw_rect12(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect12 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect12

    def _draw_rect13(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect13 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect13

    def _draw_rect14(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect14 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect14

    def _draw_rect15(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect15 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect15

    def _draw_rect16(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [UNIT * x + padding, UNIT * y + padding, UNIT * (x + 1) - padding,
                UNIT * (y + 1) - padding]
        self._draw_rect16 = self.canvas.create_rectangle(*coor, fill=color)
        return self._draw_rect16

    # 创建迷宫
    def build_maze(self):
        # 创建画布 Canvas
        self.canvas = tk.Canvas(self, bg='white',
                                width=WIDTH * UNIT,
                                height=HEIGHT * UNIT)

        # 绘制横纵方格线
        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 零点（左上角）
        origin = np.array([20, 20])

        # 创建我们的探索者 机器人（robot）
        robot_center = origin + np.array([UNIT * 3, UNIT * 8])
        self.robot = self.canvas.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')

        # 宝藏
        treasure_center = origin + np.array([UNIT * 3, 0])
        self.treasure = self.canvas.create_rectangle(
            treasure_center[0] - 15, treasure_center[1] - 15,
            treasure_center[0] + 15, treasure_center[1] + 15,
            fill='green')

        # 设置好上面配置的场景
        self.canvas.pack()

    # 重置（游戏重新开始，将机器人放到左下角）
    def reset(self):
        self.update()
        self.canvas.delete(self.robot)  # 删去机器人
        origin = np.array([20, 20])
        robot_center = origin + np.array([UNIT * 3, UNIT * 8])
        # 重新创建机器人
        self.robot = self.canvas.create_oval(
            robot_center[0] - 15, robot_center[1] - 15,
            robot_center[0] + 15, robot_center[1] + 15,
            fill='yellow')
        # 返回 观测（observation）
        return self.canvas.coords(self.robot)

    # 走一步（机器人实施 action）
    def step(self, action):
        s = self.canvas.coords(self.robot)
        base_action = np.array([0, 0])
        if action == 0:  # 上
            print('向前')
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 右
            print('向右')
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 2:  # 左
            print('向左')
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 移动机器人
        self.canvas.move(self.robot, base_action[0], base_action[1])

        # 下一个 state
        s_ = self.canvas.coords(self.robot)

        # 奖励机制
        if s_ == self.canvas.coords(self.treasure):
            reward = 1  # 找到宝藏，奖励为 1
            done = True
            s_ = 'terminal'  # 终止
            print("找到宝藏，好棒!")
        elif s_ == self.canvas.coords(self._draw_rect1):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect2):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect3):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect4):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect5):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect10):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect11):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect12):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")
        elif s_ == self.canvas.coords(self._draw_rect13):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect14):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect15):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        elif s_ == self.canvas.coords(self._draw_rect16):
            reward = -1  # 踩到炸弹1，奖励为 -1
            done = True
            s_ = 'terminal'  # 终止
            print("炸弹 爆炸...")

        else:
            reward = 0  # 其他格子，没有奖励
            done = False

        return s_, reward, done

    # 调用 Tkinter 的 update 方法
    def render(self):

        self.update()