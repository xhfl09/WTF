# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:13:00 2022

@author: xhfl0
"""
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, glob
import matplotlib.pyplot as plt

batchsize =10;
path = os.path.dirname(os.path.abspath(__file__))

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range = 30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False, 
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(path+'/train',
                                                    target_size=(128, 128),
                                                    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255) 

test_generator = test_datagen.flow_from_directory(path+'/test',
                                                  target_size=(128,128),
                                                  batch_size=batchsize,
                                                  class_mode='categorical')
#가설 1 사진이 너무 많아서?
#지금 한 파일당 총 10개의 사진이 있다 이로 인해서 느려졌다?
# 해결방법 -> 사진의 수를 줄여보자


#가설 2 사진크기
#사진크기를 조절할때 많은 시간이 소모가 된다?
# 사진크기를 속성으로 확인해 본다.
#255 255 사이즈로 조절한다 이건

#가설 3 백그라운드 파이썬
#음?
#한번 꺼보자
                                                                                                 
n_img= train_generator.n
# labels = train_generator.classes

print(n_img)
obs_imgs, obs_labels = [], []
print("안녕?")
for i in range(n_img):
    a, b = train_generator.next()
    obs_imgs.extend(a)
    obs_labels.extend(b)
    print("!\n")

imgs = np.asarray(obs_imgs) 
labels = np.asarray(obs_labels)

print("num of labels = ", labels.shape[0])

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1) 
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imgs[i], cmap=plt.cm.binary)

plt.show()