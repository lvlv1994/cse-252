#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:29:25 2018

@author: chunyilyu
"""
#%%
from mnist import MNIST
import numpy as np
from logistic import logisticRegression
#%%
num_train = 20000
num_test = 200
#%%
mndata = MNIST('./data')
mndata.load_training()
mndata.load_testing()
#%%
train_imgs = np.asarray(mndata.train_images[:num_train][:])
train_lables = np.asarray(mndata.train_labels[:num_train][:])
test_imgs = np.asarray(mndata.test_images[:num_test][:])
test_lables = np.asarray(mndata.test_labels[:num_test][:])


#%%
'''
train_imgs_23 = train_imgs[(train_lables == 2)|(train_lables == 3)]/255
train_imgs_23 = np.insert(train_imgs_23, 0, 1, axis=1)
train_labels_23 = train_lables[(train_lables == 2)|(train_lables == 3)]
'''
train_imgs_23_raw = filter(lambda x: x[1] == 2 or x[1] == 3, zip(train_imgs,train_lables))
train_imgs_23 = np.asarray(map(lambda x: x[0], train_imgs_23_raw))
train_imgs_23 = np.insert(train_imgs_23, 0, 1, axis=1)/255.
train_labels_23 = np.asarray(map(lambda x: 1 if x[1] == 2 else 0, train_imgs_23_raw))
#%%
num_samples = len(train_imgs_23)
num_pixels = len(train_imgs_23[0])
weight = np.ones(num_pixels)
#%%
lr = logisticRegression(regu_type = 'L1',regu_lambda = 0.0,num_samples=num_samples,num_pixels=num_pixels,max_iterate=400,init_weight=weight,init_learningrate=0.001,T=2000)

#%%
weight = lr.train(train_imgs_23,train_labels_23)

#%%
lr.plot_lines()
