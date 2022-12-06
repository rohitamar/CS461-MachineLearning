import numpy as np
import random
from math import log2
import statistics as st
import copy
import matplotlib.pyplot as plt
import math
from random import randint
import time

#generates the x data for a given size
#params:
#size: the size of the dataset
def p(a, b):
    return (a / b)

#generates the corresponding y labels for a given set of x values
#params:
#X: the X dataset
#sigma: the standard deviation of the noise
def p2(a, b):
    if a == 0:
        return 0
    return p(a, b) * log2(p(a, b))

#formula to find the logistic  error
#furthermore, note that log_2(0) is undefined
#hence, we add a small delta to consider this edge case
def log_error(y_i, y_pred):
    return -y_i * np.log2(y_pred + 0.00001) - ((1 - y_i) * np.log2(1 - y_pred + 0.00001))

def generate_x_data_logist(size, bias = False):
    if bias:
        X = [np.array([1] * size)]
        for _ in range(15):
            X.append(np.array([1.0 if random.random() > 0.5 else -1 for _ in range(size)]))
        X = np.array(X)
        return X, X.T
    X = np.array([np.array([1.0 if random.random() > 0.5 else -1 for _ in range(size)]) for _ in range(15)])
    return X, X.T

def generate_y_data_logist(X, sigma, bias = False):
    if bias:
        Y = np.sign(0.9 * X[1] + (0.9 ** 2) * X[2] + (0.9 ** 3) * X[3] + (0.9 ** 4) * X[4] + (0.9 ** 5) * X[5] + np.random.normal(0, sigma, len(X[0])))
        Y[Y == -1] = 0.0
        return Y
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
        #this entire function simply computes the information gain of each feature 
        #with the y labels and finds the x_index with teh largest information gain

        #we do a safety check in the following function to ensure that 
        #all information gains are not the same
        #if they are all the same, that means we are in a child node
        #and so we stop splitting (this will be explained more when we get to that part of the code)

        #Hy in the information gain formula
        #referred to cowan's notes (not much to explain in what is being done here other than that)
        Hy = -p2(len(self.y[self.y == 0]), len(self.y)) - p2(len(self.y[self.y == 1]), len(self.y))
        
        #IG finds the maximum information gain
        #ind is just the index at which the information gain between 
        #X_ind and y is highest (we pick this index)
        IG, ind = float('-inf'), -1
        
        #enumerate through each feature and see which one has the highest informationg ain
        for i, xi in enumerate(self.x):
            xi_neg_fltr = xi == -1 #filter out the negative values
            xi_pos_fltr = xi == 1 #filter out the positive values
            
            x_neg_sz = len(xi[xi_neg_fltr]) #get the number of x_i = -1
            x_pos_sz = len(xi[xi_pos_fltr]) #get the number of x_i = 1
            
            y_neg = self.y[xi_neg_fltr] #y values that have x_i = -1
            
            #find the number of y values with x_i = -1 for y = 1 and 0
            y_neg_neg = len(y_neg[y_neg == 0])
            y_neg_pos = len(y_neg[y_neg == 1])
            
            #entropy calculation
            sm_y_neg = -p2(y_neg_neg, x_neg_sz) - p2(y_neg_pos, x_neg_sz)
            y_pos = self.y[xi_pos_fltr]
            
            #find the number of y values with x_i = 1 for y = 1 and 0
            y_pos_neg = len(y_pos[y_pos == 0])
            y_pos_pos = len(y_pos[y_pos == 1])
            
            sm_y_pos = -p2(y_pos_neg, x_pos_sz) - p2(y_pos_pos, x_pos_sz) 
            
            #add everything up and account for probabilities to get final information gain
            sm = p(x_neg_sz, len(xi)) * sm_y_neg + p(x_pos_sz, len(xi)) * sm_y_pos
            IG_i = Hy - sm
            
            #update max information gain
            if IG_i > IG:
                ind, IG = i, IG_i
        
        #return the index to split at
        return ind
    
    def split(self):

        self.ind = self.find_best_index()
        xi = self.x[self.ind]
        
        #the rest of the stuff that happens here are just splitting the dataset into the left and right regions

        #note that we do this split by considering x_i == -1 and x_i == 1
        #that's what shown below
        
        fltr_left = xi == -1
        fltr_right = xi == 1
        
        x_left = [x_i[fltr_left] for x_i in self.x]
        x_right = [x_i[fltr_right] for x_i in self.x]
    
        y_left = self.y[fltr_left]
        y_right = self.y[fltr_right]

        #recursively iterate through the rest of the trees
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
    
    logistic_train_errors = []
    logistic_test_errors = []
    for sample_size in range(1, 4999, 2):
        dt = LogisticTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, sample_size)
        err = 0
        for row_num, row in enumerate(trainX_transpose):
            y_pred = LogisticTree.predict(dt, row)
            err += log_error(trainY[row_num], y_pred)
        err = err / len(trainX_transpose)
        logistic_train_errors.append(err)

        err = 0
        for row_num, row in enumerate(testX_transpose):
            y_pred = LogisticTree.predict(dt, row)
            err += log_error(testY[row_num], y_pred)
        err = err / len(testX_transpose)
        logistic_test_errors.append(err)
    
    #below snippet finds the minimum sample size and prints it
    #note that we use y < 0.01 as the error never drops to perfect 0
    min_sample_size = 0
    for x, y in zip(range(1, 4999, 2), logistic_train_errors):
        if y < 0.01:
            min_sample_size = x
        else:
            break
    print('Minimum Sample Size: {0}'.format(min_sample_size))

    #plotting the errors
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Logistic Error on Decision Tree with varying sample size')
    plt.plot(range(1, 4999, 2), logistic_train_errors, '-b', label = 'Train Error')
    plt.plot(range(1, 4999, 2), logistic_test_errors, '-r', label = 'Test Error')
    plt.xlabel('Sample Size')
    plt.ylabel('Logistic Error')
    plt.legend()
    plt.show()
    
    return logistic_train_errors, logistic_test_errors

#sigmoid function
def F(w, x):
    return 1 / (1 + np.exp(-np.dot(w, x)))

#gradient descent problem
def question2():
    #initialize 0 for all weights (note 16 = 15 + 1, 1 for bias)
    w = np.array([0] * 16)

    #get data
    trainX, trainX_transpose = generate_x_data_logist(5000, True)
    trainY = generate_y_data_logist(trainX, 0.05, True)

    testX, testX_transpose = generate_x_data_logist(500, True)
    testY = generate_y_data_logist(testX, 0.05, True)

    #we need the time at which the program starts
    #alpha value set to 0.009
    start_time, alpha = time.time(), 0.009

    #iterations, test_errors, and train_errors are being collected
    #note that we use the iterations
    #but we convert the iterations to the time it took to get to that iteration
    #to get the time vs. error graph
    iteration, test_errors = [], [] 
    train_errors = []
    for iterations in range(120000): 
        #we go up till 120,000 iterations
        #random row in the dataset
        ind = randint(0, len(trainX_transpose) - 1)
        #gradient descent update
        tmp = alpha * (F(w, trainX_transpose[ind]) - trainY[ind])
        w = w - tmp * trainX_transpose[ind]
        #for every 1000 iterations, log the error, iterations, etc
        if iterations % 1000 == 0:
            err = 0
            for row_num, row in enumerate(testX_transpose):
                y_pred = F(w, row)
                err += log_error(testY[row_num], y_pred)
            err = err / len(testX_transpose)
            iteration.append(time.time() - start_time)
            test_errors.append(err)

            err = 0
            for row_num, row in enumerate(trainX_transpose):
                y_pred = F(w, row)
                err += log_error(trainY[row_num], y_pred)
            err = err / len(trainX_transpose)
            train_errors.append(err)
    
    #print the weights and errors
    for i, weight in enumerate(w):
        print('Weight {ind}: {wei:.2f}'.format(ind = i, wei = weight))
    print('Final Training Error: {train:.6f}'.format(train = train_errors[-1]))
    print('Final Testing Error: {test:.6f}'.format(test = test_errors[-1]))

    #graph everything
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Time vs. Logistic Error')
    plt.plot(iteration, train_errors, '-r', label = 'Train Error')
    plt.plot(iteration, test_errors, '-b', label = 'Test Error')
    plt.xlabel('Time in seconds')
    plt.ylabel('Logistic Error')
    plt.legend()
    plt.show()

question1()
question2()