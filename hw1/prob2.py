#A lot of the attached code was actually pulled from a Jupyter notebook that I was using while I was coding this project
#I compiled all the scripts I used in an organized fashion from the notebook and placed it one file
#please keep note of this as you go through the code (this is why some of the variable declarations are redeclared/global)

#imports that I used throughout my code
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

#covariance of two vectors x and y
def cov(x, y):
    #definition of cov(x, y) = E[xy] - E[x]E[y]
    return np.mean(x * y) - np.mean(x) * np.mean(y)

#correlation between two vectors x and y
def corr(x, y):
    #corr(x, y) = cov(x, y) / sqrt(var(x) * var(y))
    #var(y) == 0 --> this edge case is checked in the decision tree algorithm
    #var(y) == 0 --> implies very high correlation in that set of data
    #var(x) == 0 --> implies very low correlation in that set of data (hence we return 0 on this)
    if np.var(x) == 0:
        return 0
    return cov(x, y) / math.sqrt(np.var(x) * np.var(y))

#generate a dataset of size "size" with d number of features
#meaning that d - 5 features were superfluous
def generate_data(size, d):
    x = [] #empty list
    x.append(np.random.normal(3, 1, size)) #x_1
    x.append(np.random.normal(-2, 1, size)) #x_2
    x.append(x[0] + 2 * x[1]) #x_3
    x.append((x[1] + 2)**2) #x_4
    x.append(np.random.binomial(n=1, p=0.8, size=size)) #x_5
    for _ in range(d - 5): 
        x.append(np.random.normal(0, 0.1, size)) #x_{6} through x_d

    #compute y based on the problem statement
    def compute_y(x):
        y = 4 - 3 * x[0] * x[0] + x[2] - 0.01 * x[3] + x[1] * x[4] + np.random.normal(0, 0.1, len(x[0]))
        return y
    
    #helper function to generate the transpose method
    def transpose(temp):
        temp = np.array(temp)
        return temp.T
    
    xt = transpose(x)
    y = compute_y(x)
    
    #return the matrix of x values, transpose of this matrix, and the y values
    return x, xt, y

#superfluous default dictionary 
#used in problem 2.6
#global variable for ease of access
#will be further explained later on
superfluous = defaultdict(int)

class DecisionTree():   
    #x: x matrix stored in this node
    #y: y values stored in this node
    #d: current depth at this node
    #max_depth: the maximum allowed depth throughout the entire tree (same for all nodes)
    #min_sample_size: minimum allowed sample size per node (again, same for all nodes in tree)
    def __init__(self, x, y, d, max_depth, min_sample_size):
        #most of the constructor is just declaring these variables
        #that I mentioned above 
        self.x = x
        self.y = y
        
        self.max_depth = max_depth
        self.min_sample_size = min_sample_size
        self.depth = d 
        #this is a condition to check whether a current node is a leaf node
        #note that we have np.var(y) == 0 here
        #this is just another smaller stopping condition
        #so that we can stop splitting if the variance of y is 0
        #as there's no point in splitting a node if all y values in that node are the same
        self.child = len(self.x[0]) <= self.min_sample_size or self.depth == self.max_depth or np.var(y) == 0
        
        #ind stores the feature that we split on
        self.ind = -1
        #the threshold value we split on for that feature
        self.threshold = 0
        
        #the mean of the y values (this is used if a node is a leaf/child node)
        self.result = np.mean(y)

        #the left and right nodes of this current decision tree
        self.left, self.right = None, None

        #if a node is not a child, then you can split
        if not self.child:
            self.split()
    
    #finds the best feature to split on using the maximum correlation
    def find_best_feature(self):
        #finds the correlations of all features using the corr helper function
        all_corr = [abs(corr(xi, self.y)) for xi in self.x]
        #takes the argmax of this and returns the index
        return np.argmax(all_corr)
    
    def find_threshold_split(self, ind):
        #sorts by indices (useful to find the minimum variance)
        indices = self.x[ind].argsort()
        
        #we need to sort all the other features off the selected
        #feature, and so we use argsort to do that above
        #and then reorder each of the features and y values off that
        for i in range(len(self.x)):
            self.x[i] = self.x[i][indices]
        self.y = self.y[indices]
        
        #initial threshold/min variance values
        threshold = -1
        mn = float('inf')

        #iterate for each possible threshold value
        for i in range(len(self.y) - 1):
            #the sorting becomes useful in this part
            #allows us to assume the below filter (it wouldn't be true unless the values are sorted)
            fltr_left = self.x[ind] <= self.x[ind][i] #filter values left of the current value
            fltr_right = self.x[ind] > self.x[ind][i] #filter values right of the current value
            
            #find the sizes of the left and right subarrays
            #(we can take the sum because numpy assumes that "true" is 1, "false" is 0)
            left = np.sum(fltr_left)
            right = np.sum(fltr_right)              
            
            #find the variance of the left side and the right side
            var_left = np.var(self.y[fltr_left])

            #there is a weird runtime error that I got without this ternary operator
            #sometimes, when there are duplicates right = 0 (towards the end of the posisble indices to split on)
            #so for those, we assume that the entire variance is 0
            #in practice, this makes sense becuase it is only possible for the size of the right subarray to be 0
            #when the index we are looking at it is almost the end of the entire array
            var_right = 0 if right == 0 else np.var(self.y[fltr_right]) 

            #find the variance of left and variance of right as per Professor Cowan's notes
            err_left = left / len(self.y) * var_left
            err_right = right / len(self.y) * var_right
            
            #add this up
            err = err_left + err_right
            
            #and then check if this error is smaller than the minimum variance we have had so far
            if err < mn:
                #if so, then update the threshold as the midpoint of the current value and the value next to it
                threshold, mn = (self.x[ind][i] + self.x[ind][i + 1]) / 2, err
        return threshold #return the threshold value
    
     
    def split(self):
        #splitting the decision tree
        self.ind = self.find_best_feature() #find the best feature to split on
        if self.ind > 4: #update superfluous feature, essentially if it split on something with an index greater than 4 (we're in 0-indexing since this is arrays)
            #then update the superfluous dictionary
            superfluous[self.ind] += 1

        #find the threosld to split on
        self.threshold = self.find_threshold_split(self.ind)

        #with the optimal threshold, find the left filter and right filter
        #as we did previously
        fltr_left = self.x[self.ind] <= self.threshold
        fltr_right = self.x[self.ind] > self.threshold
        
        #here we are just partitioning the x matrix of this node into the "left" and "right" matrices
        #based on the threshold we calculated
        x_left = [arr[fltr_left] for arr in self.x]
        x_right = [arr[fltr_right] for arr in self.x]
            
        #we do the same for the y matrix
        y_left = self.y[fltr_left]
        y_right = self.y[fltr_right]
        
        #recursively call the left and right decision trees, update the depths
        #and use the apropriate left and right x matrices and y values
        self.left = DecisionTree(x_left, y_left, self.depth + 1, self.max_depth, self.min_sample_size)
        self.right = DecisionTree(x_right, y_right, self.depth + 1, self.max_depth, self.min_sample_size)

    #simple DFS based static method to predict an array of values
    #we access a node's index and see if the array's value at that index
    #is less or greater than the threshold
    #based on that we traverse left or right
    #if it's a child node, then we stop and return the value
    @staticmethod
    def predict(node, arr):
        if node.child: #meaning we are at a leaf node in the decision tree
            return node.result 
        if arr[node.ind] <= node.threshold:
            return DecisionTree.predict(node.left, arr)
        else:
            return DecisionTree.predict(node.right, arr)

#data generation for all questions related to decision tree
x_train50, tx_train50, y50 = generate_data(10000, 50)
x_test50, tx_test50, testy50 = generate_data(1000, 50)

#for the d = 10, we just take the first 10 features out of the d = 50 dataset above
x_train10 = x_train50[0:10]
x_test10 = x_test50[0:10]

#take the transpose helper method from the generate dataset
def transpose(temp):
    temp = np.array(temp)
    return temp.T

#build the appropriate matrices
tx_train10 = transpose(x_train10)
tx_test10 = transpose(x_test10)

#note that the y values dont need any changing because the y values dont depend on the superfluous features
y10 = y50
testy10 = testy50

#Note that while we have different variables representing different datasets
#In each function, I used x, xt, y, testy, tx, testy to represent the different matrices/datasets that are being used
#x: the training x matrix, each row represents an entire feature
#xt: the transpose of training x
#y: the training set of y values
#tx: the transpose of testing x (we don't use the normal "version" of testing x; the return is omitted)
#testy: the training set of y values

#Finally, before we begin, here is the compute_mse_dt function
#which simply computes the mean squared error of a decision tree
#takes in the testing x matrix, testing y values, and a decision tree (which is "dt")
def compute_mse_dt(x, y, dt):
    err = 0 #error
    for i in range(len(y)):
        yp = DecisionTree.predict(dt, x[i]) #get prediction from decision tree
        err += (abs(yp - y[i]) ** 2) #add squared error
    err = err / len(y) #take mean of squared error
    return err

#Question 1
def question1():
    #Set the y values appropriately
    y = y10
    testy = testy10
    #constant model = the mean of y values (explained in pdf)
    optimal_c = np.mean(y)
    #function that computes the MSE of a given constant model
    def compute_mse_constant_model(y, c):
        err = 0
        for i in range(len(y)):
            err += (c - y[i]) ** 2
        err = err / len(y) #take average of squared sum
        return err
    
    #Find the errors of both training and testing
    training_error = compute_mse_constant_model(y, optimal_c)
    testing_error = compute_mse_constant_model(testy, optimal_c)
    
    #Print
    print('Optimal C: ', optimal_c)
    print('Training Error: ', training_error)
    print('Testing Error: ', testing_error)

    #return what we need
    return optimal_c, training_error, testing_error

#Question 2
def question2():
    #Error lists -- used for building the graphs
    train_errors = []
    test_errors = []
    
    #As we mentioned above, set the appropriate variables for our function
    x = x_train10
    y = y10
    
    tx = tx_test10
    testy = testy10
    
    xt = tx_train10
    
    for i in range(1, 30):    
        print('Iteration: ', i)
        #provide copies of the data (we want to use the same dataset on each tree)
        dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, i, 1)
        
        #compute errors
        training_err = compute_mse_dt(xt, y, dt)
        test_err = compute_mse_dt(tx, testy, dt)
        
        train_errors.append(training_err)
        test_errors.append(test_err)
    
    #build plot (can be seen in pdf)
    plt.figure(figsize = (6, 10), dpi = 80)
    plt.title('Training/Testing MSE on Decision Tree with varying depth')
    plt.plot(train_errors, '-b', label = 'Train')
    plt.plot(test_errors, '--r', label = 'Test')
    plt.xlabel('Depth')
    plt.ylabel('Training/Testing MSE')
    plt.legend()
    plt.show()
    
    return train_errors, test_errors

train_2, test_2 = question2() #shows the graph
print('Testing Errors for increasing maximum depth: ')
for i in range(len(train_2)): #prints the training/testing errors
    print(i + 1, ' ', train_2[i], ' ', test_2[i])

#Question 3
def question3():
    #Lists for building error graphs
    train_errors = []
    test_errors = []
    
    #Again set the variables as appropriate
    x = x_train10
    y = y10
    
    tx = tx_test10
    testy = testy10
    
    xt = tx_train10
    
    #set of sample sizes that we will build decision trees out of
    #note that these are a set of decreasing and somewhat random numbers (more explanation in pdf)
    sample_sizes = [10000, 5000, 2500, 1250, 625, 300, 200, 150, 100, 50, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    #generate dataset
    #x and y are used for the decision tree
    #xt is transpose of training dataset
    #tx (transpose of test dataset) and testy are a part of testing dataset
    iteration = 0
    for sample_size in sample_sizes: #iterate through each sample size
        #build decision tree with copies of dataset
        iteration += 1 #keep track of iteration
        print('Iteration: ', iteration)
        dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, 150, sample_size)
        
        #compute and store errors
        training_err = compute_mse_dt(xt, y, dt)
        test_err = compute_mse_dt(tx, testy, dt)
        train_errors.append(training_err)
        test_errors.append(test_err)
    
    #build plots
    plt.figure(figsize = (6, 10), dpi=80)
    plt.title('Training/Testing MSE on Decision Tree with varying sample size')
    plt.plot(sample_sizes, train_errors, '-b', label = 'Train')
    plt.plot(sample_sizes, test_errors, '--r', label = 'Test')
    plt.xlabel('Sample Size')
    plt.ylabel('Training/Testing MSE')
    plt.legend()
    plt.show()
    
    return train_errors, test_errors

train_3, test_3 = question3() #call function
for i in range(len(train_3)): #printing
    print(train_3[i], ' ', test_3[i])

def question5():
    #again, set the variables as appropriate
    x = x_train50
    y = y50
    
    tx = tx_test50
    testy = testy50
    
    xt = tx_train50

    #note that there are two separate functions
    #this is just because I wanted to be able to have repeated tests of my decision trees
    #in practice, these two functions do exactly the same things as question 2.2 and question 2.3
    #besides the fact that we are just using different datasets
    #everything else is the same

    def question5_depth():
        train_errors = []
        test_errors = []
        
        for i in range(1, 30):    
            print('Iteration: ', i)
            #provide copies of the data (we want to use the same dataset on each tree)
            dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, i, 1)
            
            #compute errors
            training_err = compute_mse_dt(xt, y, dt)
            test_err = compute_mse_dt(tx, testy, dt)
            
            train_errors.append(training_err)
            test_errors.append(test_err)
        
        #build plot (can be seen in pdf)
        plt.figure(figsize = (6, 10), dpi = 80)
        plt.title('Training/Testing MSE on Decision Tree with varying depth')
        plt.plot(train_errors, '-b', label = 'Train')
        plt.plot(test_errors, '--r', label = 'Test')
        plt.xlabel('Depth')
        plt.ylabel('Training/Testing MSE')
        plt.legend()
        plt.show()
        
        return train_errors, test_errors

    def question5_sample_size():
        train_errors = []
        test_errors = []
        
        #set of sample sizes that we will build decision trees out of
        #note that these are a set of decreasing and somewhat random numbers (more explanation in pdf)
        sample_sizes = sample_sizes = [10000, 5000, 2500, 1250, 625, 300, 200, 150, 100, 50, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        iteration = 0
        for sample_size in sample_sizes: #iterate through each sample size
            #build decision tree with copies of dataset
            dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, 150, sample_size)
            iteration += 1
            print('Iteration: ', iteration)
            #compute and store errors
            training_err = compute_mse_dt(xt, y, dt)
            test_err = compute_mse_dt(tx, testy, dt)
            train_errors.append(training_err)
            test_errors.append(test_err)
        
        #build plots
        plt.figure(figsize = (6, 10), dpi=80)
        plt.title('Training/Testing MSE on Decision Tree with varying sample size')
        plt.plot(sample_sizes, train_errors, '-b', label = 'Train')
        plt.plot(sample_sizes, test_errors, '--r', label = 'Test')
        plt.xlabel('Sample Size')
        plt.ylabel('Training/Testing MSE')
        plt.legend()
        plt.show()
        
        return train_errors, test_errors
    
    train_depth, test_depth = question5_depth() #call depth function
    train_size, test_size = question5_sample_size() #call sample size function 
    return train_depth, test_depth, train_size, test_size #return everything

train_depth, test_depth, train_size, test_size = question5()

#Question 6
def question6():
    sf_features = [] #hold the superfluous features
    indices = [] #hold the indices (for building the graph)
    for i in range(6, 50): #loop through from d = 6 to d = 50
        indices.append(i) 
        x, _, y = generate_data(10000, i) #build new dataset
        #print('Number of features: ', i)
        dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, 10, 1) #build decision tree off this
        sf_features.append(len(superfluous)) #append the number of superfluous features, which is just the length of the superfluous dictionary
        #note that since "superfluous" is a dictionary, it stores the unique elements -- hence, taking the size is enough to know the number of 
        #superfluous features in the tree
        #we also check if the index is greater than 4 (because that's the definition of a superfluous feature)
        #hence, this length is accurate by the definition of a superfluous feature
        #clear the superflous global variable as we want to start with an empty dictionary each time
        superfluous.clear()
    
    #Plot/Graph building
    plt.title('"d" value vs. Superfluous features')
    plt.xlabel('d')
    plt.ylabel('Superfluous features')
    plt.plot(indices, sf_features)
    plt.show()

#there isn't much that is new to this generate function
#the only thing in particular that is new to this function is that there is an additional columns filled with 1s
#this is because we need to account for the bias term in the linear regression model
#because of this, we have to reindex the dataset
#other than that, there are no other major changes
def generate_data_lin_reg(size, d):
    x = []
    x.append(np.ones(size)) #add a column of 1's now to reflect the bias term in a linear regression model
    #the rest stays the same
    x.append(np.random.normal(3, 1, size)) #X_1 as per the problem
    x.append(np.random.normal(-2, 1, size)) #X_2
    x.append(x[1] + 2 * x[2]) #X_3
    x.append((x[2] + 2)**2) #X_4
    x.append(np.random.binomial(n=1, p=0.8, size=size)) #X_5
    for _ in range(d - 5): #if d > 5, then we add d - 5 additional normally distributed random variables
        x.append(np.random.normal(0, 0.1, size))

    def compute_y(x): #use this helper function to compute the y value based on the model in the problem statement
        y = 4 - 3 * x[1] * x[1] + x[3] - 0.01 * x[4] + x[2] * x[5] + np.random.normal(0, 0.1, len(x[0]))
        return y
    
    def transpose(temp): #a helper method to compute the transpose of the list (also converts to numpy list)
        temp = np.array(temp)
        return temp.T
    
    xt = transpose(x)
    y = compute_y(x)
    
    return np.array(x), xt, y 

#there is a new error function now
#the error function works in terms of the weights of the linear regression model
#similar to how we did the decision tree model
def compute_mse_lin_reg(x, y, w):
    err = 0 #err
    for i in range(len(x)):
        err += abs(np.matmul(w, x[i]) - y[i]) ** 2 #compute the squared error
    err = err / len(x) #mean of the errors
    return err

#gradient descent implementation
def gradient_descent(x, xt, y, testx, testy):
    w = [1] * len(xt) #start with an arbitrary set of weights -- I opted for a vector filled with 1s
    xtx = np.matmul(xt, x) #multiply xt and x to get x^t x
    
    #to find alpha, find a value such that it is greater than 0 but less than 2/(max eigenvalue of xtx)
    #1 / (max eigenvalue of xtx) always follows the above, so I use that each time
    def find_alpha(xtx): 
        eigvals = np.linalg.eigvals(xtx)
        mx = np.max(eigvals)
        return 1 / mx
    
    alpha = find_alpha(xtx) #store this value 
    grad = np.matmul(xtx, w) - np.matmul(xt, y) #find the initial gradient
    
    #stopping condition is if the gradient is less than 0.1
    #note that in the bonus, I changed this to 2
    #I add this as a comment and didn't make a separate function because the number 2 is somewhat random
    #and I actually changed that because my computer couldn't handle 0.1 for some reason (too many iterations)
    #On a better computer, I think the 0.1 would still hold
    #I discuss more about this on the pdf
    while np.linalg.norm(grad) > 0.1:
        grad = np.matmul(xtx, w) - np.matmul(xt, y)
        w = w - alpha * grad 
    
    train_err = compute_mse_lin_reg(x, y, w) #training_error
    test_err = compute_mse_lin_reg(testx, testy, w) #test_error
    
    return w, train_err, test_err

xt, x, y = generate_data_lin_reg(10000, 10) #get our linear regression training dataset
_, testx, testy = generate_data_lin_reg(1000, 10) #our testing linear regression dataset
w, train_err, test_err = gradient_descent(x, xt, y, testx, testy) #apply gradient descent

#print weights, training error, and testing error
print('Weights: ')
print(w)
print('Training Error: ', train_err)
print('Testing Error: ', test_err)

#Question 8
#Build a new dataset since it's now d = 50
xt, x, y = generate_data_lin_reg(10000, 50) #training
_, testx, testy = generate_data_lin_reg(1000, 50) #testing
w, train_err, test_err = gradient_descent(x, xt, y, testx, testy) #apply gradient descent

def question8(xt, y, testx, testy, w):
    espss = np.linspace(0, 10, num = 50) #this will be our array of epsilons that we will conside r
    test_errors = [] #for graph, testing errors and number of superfluous features
    superfluous = [] #superflous features
    
    testx = testx.T #take transpose of testx matrix
    
    for eps in espss: #try each epsilon
        wcpy = copy.deepcopy(w) #make a copy of w since we don't want to update w itself
        fltr = abs(wcpy) > eps #filter out the ones that are greater than it and retrain the entire model (using gradient descent) with the new weights
        
        superfluous.append(fltr[6:].sum()) #we can simply append the number of trues to superfluous
        #note that we slice after the 5th element as beyond that, every feature is a superfluous feature

        #build the new dataset (some numpy tricks to quickly pull all elements from the filter) 
        xt_new = xt[fltr]
        testx_new = testx[fltr]
        testx_new = testx_new.T
        
        x_new = xt_new.T
        
        #pass it into gradient descent and then retrieve the testing error
        _, _, test_err = gradient_descent(x_new, xt_new, y, testx_new, testy)
        test_errors.append(test_err)
    
    #Graph building
    plt.title('Epsilon and Testing Dataset Error')
    plt.xlabel('Epsilon')
    plt.ylabel('Test Error')
    plt.plot(espss, test_errors)
    plt.show()
    
    plt.title('Epsilon and Superfluous features')
    plt.xlabel('Epsilon')
    plt.ylabel('Superfluous features')
    plt.plot(espss, superfluous)
    plt.show()
    
    return espss, test_errors, superfluous

#function call    
epss, _, sf = question8(xt, y, testx, testy, w)

#BONUS QUESTION
#we now need to add the quadratic terms to our decision tree dataset and the linear regression dataset
def generate_data_lin_reg_bonus(size):
    x = []
    x.append(np.ones(size)) #add a column of 1's now to reflect the bias term in a linear regression model
    #the rest stays the same
    x.append(np.random.normal(3, 1, size)) #X_1 as per the problem
    x.append(np.random.normal(-2, 1, size)) #X_2
    x.append(x[1] + 2 * x[2]) #X_3
    x.append((x[2] + 2)**2) #X_4
    x.append(np.random.binomial(n=1, p=0.8, size=size)) #X_5
    #quadratic terms
    for i in range(1, 7):
        for j in range(i, 7):
            x.append(x[i] * x[j])
            
    def compute_y(x): #use this helper function to compute the y value based on the model in the problem statement
        y = 4 - 3 * x[1] * x[1] + x[3] - 0.01 * x[4] + x[2] * x[5] + np.random.normal(0, 0.1, len(x[0]))
        return y
    
    def transpose(temp): #a helper method to compute the transpose of the list (also converts to numpy list)
        temp = np.array(temp)
        return temp.T
    
    xt = transpose(x)
    y = compute_y(x)
    
    return np.array(x), xt, y 

#generates the modified dataset for the bonus question (this is for the decision tree)
def generate_data_bonus(size):
    x = []
    x.append(np.random.normal(3, 1, size)) #X_1 as per the problem
    x.append(np.random.normal(-2, 1, size)) #X_2
    x.append(x[0] + 2 * x[1]) #X_3
    x.append((x[1] + 2)**2) #X_4
    x.append(np.random.binomial(n=1, p=0.8, size=size)) #X_5
    x.append(np.random.normal(0, 0.1, size))
    for i in range(0, 6):
        for j in range(0, 6):
            x.append(x[i] * x[j])
            
    def compute_y(x):
        y = 4 - 3 * x[0] * x[0] + x[2] - 0.01 * x[3] + x[1] * x[4] + np.random.normal(0, 0.1, len(x[0]))
        return y
    
    def transpose(temp): #a helper method to compute the transpose of the list (also converts to numpy list)
        temp = np.array(temp)
        return temp.T
    
    xt = transpose(x)
    y = compute_y(x)
    
    return x, xt, y

#the bonus question is just a compilation of some of the things that I have been doing previously
#we used a constant model, a linear regression model, and a decision tree
#analysis on the pdf
#but, the idea is that the linear regression overfits, and the decision tree is still better
def questionBonus():
    def compute_mse_constant_model(y, c):
        err = 0
        for i in range(len(y)):
            err += (c - y[i]) ** 2
        err = err / len(y) #take average of squared sum
        return err
    
    xt, x, y = generate_data_lin_reg_bonus(10000)
    _, testx, testy = generate_data_lin_reg_bonus(1000)
    
    optimal_c = np.mean(y)
    #Find the errors of both training and testing
    training_error = compute_mse_constant_model(y, optimal_c)
    testing_error = compute_mse_constant_model(testy, optimal_c)

    #Constant Model
    print('Training Error:', training_error)
    print('Testing Error: ', testing_error)
    
    w, train_err, test_err = gradient_descent(x, xt, y, testx, testy)

    print('Weights: ')
    print(w)
    print('Training Error: ', train_err)
    print('Testing Error: ', test_err)
    
    x, xt, y = generate_data_bonus(10000)
    _, tx, testy = generate_data_bonus(1000)
    dt = DecisionTree(copy.deepcopy(x), copy.deepcopy(y), 1, 15, 5)
    
    training_err = compute_mse_dt(xt, y, dt)
    testing_err = compute_mse_dt(tx, testy, dt)
    
    print('Training Error:', training_err)
    print('Testing Error: ', testing_err)
    
    