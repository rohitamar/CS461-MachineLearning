import numpy as np

class SoftMax:
    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size) / in_size
        self.biases = np.zeros(out_size)
        self.cache = {}
        
    def forward(self, input):
        self.cache['prev_shape'] = input.shape
        
        input = input.flatten()
        self.cache['prev_input'] = input 
        
        Z = np.dot(input, self.weights) + self.biases
        self.cache['prev_lin'] = Z
        
        A = np.exp(Z)
        return A / np.sum(A, axis = 0)
    
    def backprop(self, d_L_d_out, rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0: continue

            t_exp = np.exp(self.cache['prev_lin'])
            sm = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (sm ** 2)
            d_out_d_t[i] = t_exp[i] * (sm - t_exp[i]) / (sm ** 2)

            d_t_d_w = self.cache['prev_input']
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t

            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= rate * d_L_d_w
            self.biases -= rate * d_L_d_b
            
            return d_L_d_inputs.reshape(self.cache['prev_shape'])
