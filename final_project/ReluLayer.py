import numpy as np

class Relu:
    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size) / in_size
        self.biases = np.zeros(out_size)
        self.cache = {}
    
    def relu(x):
        return np.maximum(0, x)

    def forward(self, input):
        input = input.flatten()

        Z = np.dot(input, self.weights) + self.biases
        A = self.relu(input)
        return A
    
    