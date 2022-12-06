import numpy as np
import random 
import math

INPUT_SZ = 30

def generate_data_set(variance, size):
    X = []
    X.append(np.random.normal(0, 1, size)) #1
    X.append(X[0] + np.random.normal(0, variance ** 0.5, size)) #2
    X.append(X[0] + np.random.normal(0, variance ** 0.5, size)) #3
    for i in range(3, 30):
        X.append(X[i - 3] + np.random.normal(0, variance ** 0.5, size))
    X = np.array(X)
    X = X.T
    return X

def tanh_prime(x):
    return 1. - np.tanh(x) ** 2

X = generate_data_set(0.1, 5000)

def forward_pass_mat(i, iterations, alpha_val):
    K = i
    
    ALPHA = alpha_val

    xav = math.sqrt(6/(30 + K))

    W2 = np.random.uniform(-xav, xav, K * 30).reshape(30, K)
    W1 = np.random.uniform(-xav, xav, K * 30).reshape(K, 30)

    B2 = np.zeros(30).reshape(30, 1)
    B1 = np.zeros(K).reshape(K, 1) 
    
    for _ in range(iterations):
        r = random.randint(0, 4999)    
        X0 = X[r].reshape(-1, 1)

        Z1 = W1.dot(X0) + B1
        X1 = np.tanh(Z1)
        X2 = W2.dot(X1) + B2

        delta_2 = 2 * (X2 - X0)
        gradE_gradW_2 = delta_2.dot(X1.T)
        W2 = W2 - ALPHA * gradE_gradW_2
        B2 = B2 - ALPHA * delta_2

        W2_T = W2.T
        W2_T_times_delta_2 = W2_T.dot(delta_2)
        
        W_1_times_X0_with_bias = Z1
        f_prime = tanh_prime(W_1_times_X0_with_bias)

        delta_1 = W2_T_times_delta_2 * f_prime

        X0_T = X0.T
        gradE_gradW_1 = delta_1.dot(X0_T)
        W1 = W1 - ALPHA * gradE_gradW_1
        B1 = B1 - ALPHA * delta_1

    final_loss = 0
    for xi in X:
        X0 = xi.reshape(-1, 1)
        Z1 = (W1 @ X0) + B1
        X1 = np.tanh(Z1)
        X2 = (W2 @ X1) + B2
        diff = (X2 - X0).reshape(-1)
        final_loss += np.sum(diff * diff)
    final_loss = final_loss / 5000
    print('Final Loss at', K, ': ', final_loss)
    return final_loss

mn = float('inf')

# iterations = 0
# alpha_val = 0

# for itera in range(2000, 6500, 50):
#     for alpha in np.linspace(0.001, 0.005, num = 200):
#         tmp = forward_pass_mat(30, itera, alpha)
#         print(itera, alpha, tmp)
#         if tmp < mn:
#             mn = tmp
#             iterations, alpha_val = itera, alpha

# print('Iterations: ', iterations)
# print('Alpha: ', alpha_val)

hidden_layer_size = list(range(1, 31))
errors = [forward_pass_mat(i, 100000, 0.0018) for i in range(1, 31)]
import matplotlib.pyplot as plt
plt.xlabel('Number of Neurons in Hidden Layer (k)')
plt.ylabel('Error of the Neural Network')
plt.plot(hidden_layer_size, errors)
plt.show()