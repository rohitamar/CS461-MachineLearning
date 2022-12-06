import numpy as np

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        #Xavier Initialization
        self.kernels = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) / (kernel_size ** 2)
        self.bias = np.zeros((out_channels))
        self.cache = {}

    def generator(self, image):
        h, w = image.shape[0], image.shape[1]

        for x in range(h - self.kernel_size + 1):
            for y in range(w - self.kernel_size + 1):
                region = image[x:(x + self.kernel_size), y:(y + self.kernel_size)]
                yield region, x, y
    
    def forward(self, input):
        h, w = input.shape[0], input.shape[1]
        self.cache['prev_input'] = input
        result = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.out_channels))

        for region, x, y in self.generator(input):
            result[x, y] = np.sum(region * self.kernels, axis = (1, 2, 3)) + self.bias
        
        return result
    
    def backprop(self, d_L_d_out, rate):
        d_L_d_kernels = np.zeros(self.kernels.shape)
        d_L_d_bias = np.zeros(self.out_channels)
        for region, i, j in self.generator(self.cache['prev_input']):
            for f in range(self.out_channels):
                d_L_d_bias[f] += d_L_d_out[i, j, f]
                d_L_d_kernels[f] += d_L_d_out[i, j, f] * region

        self.kernels -= rate * d_L_d_kernels
        self.bias -= rate * d_L_d_bias
