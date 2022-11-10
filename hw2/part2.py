import numpy as np
import random
from math import log2
import statistics as st
import copy
import matplotlib.pyplot as plt
import math
from random import randint
import time

def p(a, b):
    return (a / b)

def p2(a, b):
    if a == 0:
        return 0
    return p(a, b) * log2(p(a, b))

def log_error(y_i, y_pred):
    return -y_i * np.log2(y_pred + 0.00001) - ((1 - y_i) * np.log2(1 - y_pred + 0.00001))

def generate_x_data_logist(size):
    X = np.array([np.array([1.0 if random.random() > 0.5 else -1 for _ in range(size)]) for _ in range(15)])
    return X, X.T

def generate_y_data_logist(X, sigma):
    Y = np.sign(0.9 * X[0] + (0.9 ** 2) * X[1] + (0.9 ** 3) * X[2] + (0.9 ** 4) * X[3] + (0.9 ** 5) * X[4] + np.random.normal(0, sigma, len(X[0])))
    Y[Y == -1] = 0.0
    return Y

class LogisticTree():
    def __init__(self, x, y, d, sample_size):
        
        self.x, self.y, self.d = x, y, d
        
        self.ind = -1
        self.left, self.right = None, None
        
        self.child = len(self.y) <= sample_size or d >= 1000 or np.all(self.y == self.y[0])
        self.result = ((self.y == 1).sum()) / len(self.y)
        
        self.sample_size = sample_size
        
        if not self.child:
            self.split()

    def find_best_index(self):
        Hy = -p2(len(self.y[self.y == 0]), len(self.y)) - p2(len(self.y[self.y == 1]), len(self.y))
        
        IG, ind = float('-inf'), -1
        
        for i, xi in enumerate(self.x):
            xi_neg_fltr = xi == -1
            xi_pos_fltr = xi == 1
            
            x_neg_sz = len(xi[xi_neg_fltr])
            x_pos_sz = len(xi[xi_pos_fltr])
            
            y_neg = self.y[xi_neg_fltr]
            
            y_neg_neg = len(y_neg[y_neg == 0])
            y_neg_pos = len(y_neg[y_neg == 1])
            
            sm_y_neg = -p2(y_neg_neg, x_neg_sz) - p2(y_neg_pos, x_neg_sz)
            y_pos = self.y[xi_pos_fltr]
            
            y_pos_neg = len(y_pos[y_pos == 0])
            y_pos_pos = len(y_pos[y_pos == 1])
            
            sm_y_pos = -p2(y_pos_neg, x_pos_sz) - p2(y_pos_pos, x_pos_sz) 
            
            sm = p(x_neg_sz, len(xi)) * sm_y_neg + p(x_pos_sz, len(xi)) * sm_y_pos
            IG_i = Hy - sm
            
            if IG_i > IG:
                ind, IG = i, IG_i
        
        #print(" ", ind)
        return ind
    
    def split(self):
        self.ind = self.find_best_index()
        xi = self.x[self.ind]
        
        fltr_left = xi == -1
        fltr_right = xi == 1
        
        x_left = [x_i[fltr_left] for x_i in self.x]
        x_right = [x_i[fltr_right] for x_i in self.x]
    
        y_left = self.y[fltr_left]
        y_right = self.y[fltr_right]

        self.left = LogisticTree(x_left, y_left, self.d + 1, self.sample_size)
        self.right = LogisticTree(x_right, y_right, self.d + 1, self.sample_size)
    
    @staticmethod
    def predict(node, arr):
        if node.child:
            return node.result
        return LogisticTree.predict(node.left, arr) if arr[node.ind] == -1 else LogisticTree.predict(node.right, arr)

def question1():
    trainX, trainX_transpose = generate_x_data_logist(5000)
    trainY = generate_y_data_logist(trainX, 0.05)
    
    testX, testX_transpose = generate_x_data_logist(500)
    testY = generate_y_data_logist(testX, 0.05)
    
    logistic_errors = []
    for sample_size in range(1, 501, 2):
        dt = LogisticTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, sample_size)
        err = 0
        for row_num, row in enumerate(trainX_transpose):
            y_pred = LogisticTree.predict(dt, row)
            err += log_error(trainY[row_num], y_pred)
        err = err / len(trainX_transpose)
        logistic_errors.append(err)
    
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Logistic Error on Decision Tree with varying sample size')
    plt.plot(range(1, 501, 2), logistic_errors, '-b', label = 'Train Error')
    #plt.plot(range(1, 5001, 2), test_mismatches, '-r', label = 'Test Error')
    plt.xlabel('Sample Size')
    plt.ylabel('Logistic Error')
    plt.legend()
    plt.show()
    
    return logistic_errors

def F(w, x):
    return 1 / (1 + math.exp(-np.dot(w, x)))

def question2():
    w = np.array([1] * 15)

    trainX, trainX_transpose = generate_x_data_logist(5000)
    trainY = generate_y_data_logist(trainX, 0.05)
    
    testX, testX_transpose = generate_x_data_logist(500)
    testY = generate_y_data_logist(testX, 0.05)

    start_time = time.time()
    alpha = 0.2

    # while time.time() - start_time < 50:
    #     ind = randint(0, len(trainX_transpose) - 1)
    #     tmp = alpha * (F(w, trainX_transpose[ind]) - trainY[ind])
    #     w = w - tmp * trainX_transpose[ind]

    err = 0
    for row_num, row in enumerate(testX_transpose):
        y_pred = F(w, row)
        err += log_error(testY[row_num], y_pred)
    
    err = err / 500
    print('Error: ', err)
question2()