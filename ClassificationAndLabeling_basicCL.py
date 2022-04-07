# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:15:51 2022

@author: xhfl0
"""


import cv2 #이미지 처리하는 라이버리 패키지
import os, glob 
import numpy
import matplotlib.pyplot as plt 

image_size = (128,128) # 학습을 위해 이미지 사이즈 조절
image_folder = 5; 
#__file__ = 'ClassfiationAndLableing_bassicCL.py'

path = os.path.dirname(os.path.abspath(__file__)) #fil이 들어있는 곳의 이름

obs_Xdata = []
obs_Ylabel = []
path_obs = []
#obs_xdata 가 이미지 저장하는 공간
#그것의 레이블은 obs_Ylabe에 있다
# 각 폴더5개 있는 사진을
for i in range(0,image_folder):
    path_obs = path+'/obs'+str(i)
    obs_img_files = glob.glob(path_obs+'/*.jpg')
    #print('path =',obs_img_files)
    for j in obs_img_files:
        obs_img = cv2.imread(j)
        obs_img = cv2.resize(obs_img,image_size)
        obs_Xdata.append(obs_img)
        obs_Ylabel.append(i)
    print(obs_Xdata)
        
for j in range(0,10):
    plt.figure()
    plt.imshow(obs_Xdata[j])
    plt.show()
    
print('image label =',obs_Ylabel)
