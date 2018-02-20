#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:19:17 2018

@author: chunyilyu
"""

'''
这段code展示了Logistic Regression的梯度下降实现，这是machine learning领域十分基础的课程，虽说之后也手动实现了很多更复杂的算法，但是这段算法对我日后的理解十分有帮助，我把他写成了一个包，可以直接调用，也体现了python面向对象的编程
'''
import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self,regu_type,regu_lambda,num_samples,num_pixels,max_iterate,init_weight,init_learningrate,T):
        self.regu_type = regu_type
        self.regu_lambda = regu_lambda
        self.num_samples = int(0.9*num_samples)
        self.num_pixels = num_pixels
        self.max_iterate = max_iterate
        self.init_weight = init_weight
        self.init_learningrate = init_learningrate
        self.T = T
        self.weight = init_weight
        self.train_loss = []
        self.holdon_loss = []
        self.train_error = []
        self.holdon_error = []
        self.weight_record = []
    # SIGMOID Function
    def _sigmoid_function(self,weight,X):
        return 1./float(1+np.exp(-np.dot(X,weight)))
    # Compute gradient to get find the next optimazation step
    def _get_gradient(self,X,y,weight):
        
        derivate = np.zeros(self.num_pixels)
        for i in range(self.num_samples):
            
            derivate -= np.dot((y[i] - self._sigmoid_function(weight,X[i])),X[i])
        if self.regu_type == 'L1':
            derivate += np.sign(weight) * self.regu_lambda
        elif self.regu_type == 'L2':
            derivate += 2 * weight * self.regu_lambda        
        return derivate
    # Comput loss function to optimaze module
    def loss_function(self,X,y,weight):
        loss = 0.0
        for i in xrange(len(y)):            #Take different value if label is different
            if y[i]:
                loss -= np.log(self._sigmoid_function(weight,X[i]))
            else:
                loss -= np.log(1. / (1+ np.exp(np.dot(X[i], weight))))  
        loss /= self.num_samples
        
        if self.regu_type == 'L1':
            loss += self.regu_lambda * np.linalg.norm(weight, ord = 1)
        else:
            loss += self.regu_lambda * np.dot(weight, weight)
        print(loss)
   
        return loss

    # Evaluate the module's performance
    def percent_correct(self,X,y,weight):
        correct = 0
        for i in range(len(y)):
            y_hat = self._sigmoid_function(weight,X[i])
            if y_hat > 0.5 and y[i]:
                correct += 1
            elif y_hat < 0.5 and y[i] == 0:
                correct += 1
        return correct/self.num_samples
    #Train our model
    def train(self,X,y):
        weight = self.init_weight
       
        training_X = X[:int(self.num_samples)]
        holdon_X = X[int(self.num_samples):]
        training_y = y[:int(self.num_samples)]
        holdon_y = y[int(self.num_samples):]  
        
        # start iteration here
        for i in range(self.max_iterate):
            stepSize = self.init_learningrate/ (1 + np.float(i/self.T))
            derivative = self._get_gradient(training_X,training_y,weight) #To get next step
            weight -= stepSize * derivative                #Re-evaluate the weight
            self.weight_record.append(weight[:])
            train_loss = self.loss_function(training_X,training_y,weight)
            holdon_loss = self.loss_function(holdon_X,holdon_y,weight)
            self.train_loss.append(train_loss)
            self.holdon_loss.append(holdon_loss)            # Recorde results
            
            train_error = self.percent_correct(training_X,training_y,weight)
            holdon_error = self.percent_correct(holdon_X,holdon_y,weight)
            
            self.train_error.append(train_error)
            self.holdon_error.append(holdon_error)
            
            
            #early stoping
            if len(self.holdon_error) > 30 and self.holdon_error[-1] > self.holdon_error[-2] and self.holdon_error[-2] > self.holdon_error[-3]:
                weight = self.weight_record[-3]
                break
        self.weight = weight
        #print(self.train_loss)
        return self.weight

    def plot_lines(self):
        xaxis = [i for i in range(len(self.train_loss))]
        plt.plot(xaxis, self.train_loss)
        plt.show()
        
        
