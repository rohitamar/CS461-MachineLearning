#necessary imports
import numpy as np
import random
from math import log2
import statistics as st
import matplotlib.pyplot as plt
import copy
import sys

#generates the x data for a given size
#params:
#size: the size of the dataset
def generate_x_data(size):
    X = np.array([np.array([1 if random.random() > 0.5 else -1 for _ in range(size)]) for _ in range(15)])
    return X, X.T

#generates the corresponding y labels for a given set of x values
#params:
#X: the X dataset
#sigma: the standard deviation of the noise
def generate_y_data(X, sigma):
    Y = np.sign(0.9 * X[0] + (0.9 ** 2) * X[1] + (0.9 ** 3) * X[2] + (0.9 ** 4) * X[3] + (0.9 ** 5) * X[4] + np.random.normal(0, sigma, len(X[0])))
    return Y

#generates the x data for a given size
#params:
#size: the size of the dataset
#bias: adds an array of 1's to consider the bias
#note that bias has a default paramter of False, because most of the time you won't need this bias
#furthermore, there is no difference between generate_x_data_logist and generate_x_data
#it was just made to distinguish between the two datasets
def generate_x_data_logist(size, bias = False):
    if bias:
        X = [np.array([1] * size)]
        for _ in range(15):
            X.append(np.array([1.0 if random.random() > 0.5 else -1 for _ in range(size)]))
        X = np.array(X)
        return X, X.T
    X = np.array([np.array([1.0 if random.random() > 0.5 else -1 for _ in range(size)]) for _ in range(15)])
    return X, X.T

#generates the y labels for a given set of x values
#size: the size of the dataset
#bias: default paramter False, if bias is True
#then the calculation of the Y label is slightly different (all X columns are translated by 1)
def generate_y_data_logist(X, sigma, bias = False):
    if bias:
        Y = np.sign(0.9 * X[1] + (0.9 ** 2) * X[2] + (0.9 ** 3) * X[3] + (0.9 ** 4) * X[4] + (0.9 ** 5) * X[5] + np.random.normal(0, sigma, len(X[0])))
        Y[Y == -1] = 0.0
        return Y
    Y = np.sign(0.9 * X[0] + (0.9 ** 2) * X[1] + (0.9 ** 3) * X[2] + (0.9 ** 4) * X[3] + (0.9 ** 5) * X[4] + np.random.normal(0, sigma, len(X[0])))
    Y[Y == -1] = 0.0
    return Y

#a / b (probability of some event if we consider b to be the whole set and a the set of consideration)
def p(a, b):
    return (a / b)

#multiples the probability with the log of the probability
#calculates individual entropy of an event in the decision tree classifier
def p2(a, b):
    if a == 0:
        return 0
    return p(a, b) * log2(p(a, b))

#Decision Tree Classifier
class DecisionTree():
    cnt = 0 #cnt used to calculate number of irrelevant features (1.4)
    def __init__(self, x, y, d, sample_size): #constructor
        self.x, self.y, self.d = x, y, d #x dataset, y dataset, and depth of this node
        #note that x and y are not the entire dataset
        # they are just the partition of data that is held in that node
        self.ind = -1 #x_{ind} is the feature we will split the dataset on
        self.left, self.right = None, None #left and right nodes of the tree
        self.child = len(self.y) <= sample_size or np.all(self.y == self.y[0])
        #these two conditions check whether the given node should be a child
        #the latter condition is a safety check
        #there is no point in splitting if all the y values are the same
        #and that's what np.all does
        self.result = self.y[0] #this just stores the result 
        #note that if we are in a child node
        #then all the y values must be the same
        #so we can just pick the first value
        self.sample_size = sample_size #the sample size (in other words, the sample size threshold)
        if not self.child: #split if not a child node
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
        Hy = -p2(len(self.y[self.y == -1]), len(self.y)) - p2(len(self.y[self.y == 1]), len(self.y)) - p2(len(self.y[self.y == 0]), len(self.y))
        
        #IG finds the maximum information gain
        #ind is just the index at which the information gain between 
        #X_ind and y is highest (we pick this index)
        IG, ind = float('-inf'), -1
        
        #enumerate through each feature and see which one has the highest informationg ain
        for i, xi in enumerate(self.x):
            xi_neg_fltr = xi == -1 #filter out the negative values
            xi_pos_fltr = xi == 1 #and the positive values
            
            x_neg_sz = len(xi[xi_neg_fltr]) #get the number of x_i = -1
            x_pos_sz = len(xi[xi_pos_fltr]) #get the number of x_i = 1
            
            y_neg = self.y[xi_neg_fltr] #take the y values that have x_i = -1
            
            #find the number of y values with x_i = -1 for y = -1, 1, and 0
            y_neg_neg = len(y_neg[y_neg == -1]) 
            y_neg_pos = len(y_neg[y_neg == 1])
            y_neg_zer = len(y_neg[y_neg == 0])
            
            #use that to find the entropy
            sm_y_neg = -p2(y_neg_neg, x_neg_sz) - p2(y_neg_pos, x_neg_sz) - p2(y_neg_zer, x_neg_sz)

            #do the same but now for a positive value of x_i
            y_pos = self.y[xi_pos_fltr]
            
            y_pos_neg = len(y_pos[y_pos == -1])
            y_pos_pos = len(y_pos[y_pos == 1])
            y_pos_zer = len(y_pos[y_pos == 0])
            
            sm_y_pos = -p2(y_pos_neg, x_pos_sz) - p2(y_pos_pos, x_pos_sz) - p2(y_pos_zer, x_pos_sz)
            
            #put this all together to get the final information gain
            sm = p(x_neg_sz, len(xi)) * sm_y_neg + p(x_pos_sz, len(xi)) * sm_y_pos
            IG_i = Hy - sm

            #update the maximum information gain
            if IG_i > IG:
                ind, IG = i, IG_i
        
        #return the index and the information gain
        return ind, IG
    
    def split(self):
        #note that if the information gain is 0, then all x_i and y have
        #split in the past and so there's no point in going further
        #in fact, it actually infinitely recurses (because it won't split and so the sample_size doesn't drop)
        #thus, we stop splitting here
        self.ind, IG = self.find_best_index()
        if IG == 0.0:
            self.child = True
            return
        
        #do an update to our irrelevant features count if the index of the split is greater than equal to 5
        #note that 0 through 4 indices are all relevant features
        if self.ind >= 5:
            DecisionTree.cnt += 1
        
        #the rest of the stuff that happens here are just splitting the dataset into the left and right regions

        #note that we do this split by considering x_i == -1 and x_i == 1
        #that's what shown below
        xi = self.x[self.ind]
        
        fltr_left = xi == -1
        fltr_right = xi == 1
        
        x_left = [x_i[fltr_left] for x_i in self.x]
        x_right = [x_i[fltr_right] for x_i in self.x]
    
        y_left = self.y[fltr_left]
        y_right = self.y[fltr_right]

        #DFS and recurse further down the tree
        self.left = DecisionTree(x_left, y_left, self.d + 1, self.sample_size)
        self.right = DecisionTree(x_right, y_right, self.d + 1, self.sample_size)
    
    #general predict method
    #given an "arr" (list of values), we predict it's class
    @staticmethod
    def predict(node, arr):
        if node.child:
            return node.result
        return DecisionTree.predict(node.left, arr) if arr[node.ind] == -1 else DecisionTree.predict(node.right, arr)

    #helper method that finds the height of the decision tree
    #helpful to see how far the decision tree was recursing while developing
    @staticmethod
    def height(node):
        if node.child:
            return 1
        return 1 + max(DecisionTree.height(node.left), DecisionTree.height(node.right))


def question1():
    #train and testing error lists
    train_mismatches = []
    test_mismatches = []
    
    #getting the data
    trainX, trainX_transpose = generate_x_data(5000)
    trainY = generate_y_data(trainX, 0.05)
    
    testX, testX_transpose = generate_x_data(500)
    testY = generate_y_data(testX, 0.05)
    
    #iterating through different sample sizes
    for sample_size in range(1, 5001, 2):
        #creating the model
        dt = DecisionTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, sample_size)
        cnt = 0 #keeping count of training mismatches
        for row_num, row in enumerate(trainX_transpose):
            if DecisionTree.predict(dt, row) != trainY[row_num]: #mismatch
                cnt += 1
        train_mismatches.append(cnt)
        
        cnt = 0 #keeping count of training mismatches
        for row_num, row in enumerate(testX_transpose):
            if DecisionTree.predict(dt, row) != testY[row_num]: #mismatch
                cnt += 1
        test_mismatches.append(cnt)
    
    #below snippet finds the minimum sample size and prints it
    min_sample_size = 0
    for x, y in zip(range(1, 501, 2), train_mismatches):
        if y == 0:
            min_sample_size = x
        else:
            break
    print('Minimum Sample Size: {0}'.format(min_sample_size))

    #plotting the errors
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Error on Decision Tree with varying sample size')
    plt.plot(range(1, 5001, 2), train_mismatches, '-b', label = 'Train Error')
    plt.plot(range(1, 5001, 2), test_mismatches, '-r', label = 'Test Error')
    plt.xlabel('Sample Size')
    plt.ylabel('Number of mismatches')
    plt.legend()
    plt.show()
    
    return train_mismatches, test_mismatches

def question3():
    #129 is optimal sample size
    #train and test mismatches
    train_mismatches = []
    test_mismatches = []
    
    #getting the data
    trainX, trainX_transpose = generate_x_data(5000)
    testX, testX_transpose = generate_x_data(500)
    for noise in np.arange(0.01, 2, 0.1):
        #build the training and testing data
        trainY = generate_y_data(copy.deepcopy(trainX), noise)
        testY = generate_y_data(copy.deepcopy(testX), noise)
        
        #model
        dt = DecisionTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, 123)
        cnt = 0
        for row_num, row in enumerate(trainX_transpose):
            if DecisionTree.predict(dt, row) != trainY[row_num]:
                cnt += 1
        train_mismatches.append(cnt)
        
        cnt = 0
        for row_num, row in enumerate(testX_transpose):
            if DecisionTree.predict(dt, row) != testY[row_num]:
                cnt += 1
        test_mismatches.append(cnt)
    
    #plotting the graph
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Error on Decision Tree with varying noise')
    plt.plot(np.arange(0.01, 2, 0.1), train_mismatches, '-b', label = 'Train Error')
    plt.plot(np.arange(0.01, 2, 0.1), test_mismatches, '-r', label = 'Test Error')
    plt.xlabel('Noise (sigma value)')
    plt.ylabel('Number of mismatches')
    plt.legend()
    plt.show()

def question4():
    #129 is optimal sample size (I discuss a key point about the sample size for question 4)
    #refer to the pdf
    arr = []
    trainX, _ = generate_x_data(5000)
    testX, _ = generate_x_data(500)

    for noise in np.arange(0.01, 2, 0.05):
        trainY = generate_y_data(copy.deepcopy(trainX), noise)
        testY = generate_y_data(copy.deepcopy(testX), noise)
        dt = DecisionTree(copy.deepcopy(trainX), copy.deepcopy(trainY), 1, 123)
        arr.append(DecisionTree.cnt) #append to a list (to eventually graph it)
        DecisionTree.cnt = 0 #update the count of irrelevant features
        #note that it is a static/class variable
    
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing Logistic Error on Decision Tree with varying sample size')
    plt.plot(np.arange(0.01, 2, 0.05), arr, '-b', label = 'Irrelevant Features')
    plt.xlabel('Noise (standard deviation)')
    plt.ylabel('Irrelevant Features')
    plt.legend()
    plt.show()

question4()
