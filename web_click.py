
#coding:utf-8
from selenium import webdriver
import time

driver = webdriver.Chrome(executable_path='/home/tianzhen/Downloads/chromedriver')
driver.get('http://192.168.168.100:9090/')
#driver.maximize_window()
driver.find_element_by_xpath("./*//button[@id='VU']").click()
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

driver = webdriver.Firefox()
driver.implicitly_wait(10)
driver.maximize_window()
driver.get('http://sahitest.com/demo/clicks.htm')

click_btn = driver.find_element_by_xpath('//input[@value="click me"]')  # 单击按钮
doubleclick_btn = driver.find_element_by_xpath('//input[@value="dbl click me"]')  # 双击按钮
rightclick_btn = driver.find_element_by_xpath('//input[@value="right click me"]')  # 右键单击按钮

menu = driver.find_element_by_css_selector(".nav")
hidden_submenu = driver.find_element_by_css_selector(".nav #submenu1")

actions = ActionChains(driver)
actions.move_to_element(menu)
actions.click(hidden_submenu)
actions.perform()

menu = driver.find_element_by_css_selector(".nav")
hidden_submenu = driver.find_element_by_css_selector(".nav #submenu1")

actions = ActionChains(driver)
actions.move_to_element(menu)
actions.click(hidden_submenu)
actions.perform()
