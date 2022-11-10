import numpy as np
import random
from math import log2
import statistics as st
import matplotlib.pyplot as plt
import copy
import sys

def generate_x_data(size):
    X = np.array([np.array([1 if random.random() > 0.5 else -1 for _ in range(size)]) for _ in range(15)])
    return X, X.T

def generate_y_data(X, sigma):
    Y = np.sign(0.9 * X[0] + (0.9 ** 2) * X[1] + (0.9 ** 3) * X[2] + (0.9 ** 4) * X[3] + (0.9 ** 5) * X[4] + np.random.normal(0, sigma, len(X[0])))
    return Y

def p(a, b):
    return (a / b)

def p2(a, b):
    if a == 0:
        return 0
    return p(a, b) * log2(p(a, b))

class DecisionTree():
    cnt = 0
    def __init__(self, x, y, d, sample_size):
        self.x, self.y, self.d = x, y, d       
        self.ind = -1
        self.left, self.right = None, None
        self.child = len(self.y) <= sample_size or np.all(self.y == self.y[0])

        self.sample_size = sample_size
        
        if not self.child:
            self.split()

    def find_best_index(self):
        Hy = -p2(len(self.y[self.y == -1]), len(self.y)) - p2(len(self.y[self.y == 1]), len(self.y)) - p2(len(self.y[self.y == 0]), len(self.y))
        
        IG, ind = float('-inf'), -1
        
        for i, xi in enumerate(self.x):
            xi_neg_fltr = xi == -1
            xi_pos_fltr = xi == 1
            
            x_neg_sz = len(xi[xi_neg_fltr])
            x_pos_sz = len(xi[xi_pos_fltr])
            
            y_neg = self.y[xi_neg_fltr]
            
            y_neg_neg = len(y_neg[y_neg == -1])
            y_neg_pos = len(y_neg[y_neg == 1])
            y_neg_zer = len(y_neg[y_neg == 0])
            
            sm_y_neg = -p2(y_neg_neg, x_neg_sz) - p2(y_neg_pos, x_neg_sz) - p2(y_neg_zer, x_neg_sz)

            y_pos = self.y[xi_pos_fltr]
            
            y_pos_neg = len(y_pos[y_pos == -1])
            y_pos_pos = len(y_pos[y_pos == 1])
            y_pos_zer = len(y_pos[y_pos == 0])
            
            sm_y_pos = -p2(y_pos_neg, x_pos_sz) - p2(y_pos_pos, x_pos_sz) - p2(y_pos_zer, x_pos_sz)
            
            sm = p(x_neg_sz, len(xi)) * sm_y_neg + p(x_pos_sz, len(xi)) * sm_y_pos
            IG_i = Hy - sm
            if IG_i > IG:
                ind, IG = i, IG_i
        
        return ind, IG
    
    def split(self):

        self.ind, IG = self.find_best_index()
        if IG == 0.0:
            self.child = True
            return
        if self.ind >= 5:
            DecisionTree.cnt += 1
        xi = self.x[self.ind]
        
        fltr_left = xi == -1
        fltr_right = xi == 1
        
        x_left = [x_i[fltr_left] for x_i in self.x]
        x_right = [x_i[fltr_right] for x_i in self.x]
    
        y_left = self.y[fltr_left]
        y_right = self.y[fltr_right]

        self.left = DecisionTree(x_left, y_left, self.d + 1, self.sample_size)
        self.right = DecisionTree(x_right, y_right, self.d + 1, self.sample_size)
    
    @staticmethod
    def predict(node, arr):
        if node.child:
            return node.result
        return DecisionTree.predict(node.left, arr) if arr[node.ind] == -1 else DecisionTree.predict(node.right, arr)

    @staticmethod
    def height(node):
        if node.child:
            return 1
        return 1 + max(DecisionTree.height(node.left), DecisionTree.height(node.right))

import sys

def question4():

    arr = []
    trainX, _ = generate_x_data(5000)
    testX, _ = generate_x_data(500)

    for noise in np.arange(0.01, 2, 0.05):
        trainY = generate_y_data(copy.deepcopy(trainX), noise)
        testY = generate_y_data(copy.deepcopy(testX), noise)
        dt = DecisionTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, 3)
        arr.append(DecisionTree.cnt)
        DecisionTree.cnt = 0
        print(DecisionTree.height(dt))
    
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Logistic Error on Decision Tree with varying sample size')
    plt.plot(np.arange(0.01, 2, 0.05), arr, '-b', label = 'Irrelevant Features')
    #plt.plot(range(1, 5001, 2), test_mismatches, '-r', label = 'Test Error')
    plt.xlabel('Noise (standard deviation)')
    plt.ylabel('Irrelevant Features')
    plt.legend()
    plt.show()

question4()