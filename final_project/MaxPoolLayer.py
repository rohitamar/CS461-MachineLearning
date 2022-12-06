import numpy as np

class MaxPool:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.cache = {}

    def generate(self, image):
        h, w, _ = image.shape

        for x in range(h // self.kernel_size):
            for y in range(w // self.kernel_size):
                sx, ex = x * self.kernel_size, self.kernel_size * x + self.kernel_size
                sy, ey = y * self.kernel_size, self.kernel_size * y + self.kernel_size
                region = image[sx:ex, sy:ey]
                yield region, x, y
    
    def forward(self, input):
        self.cache['prev_input'] = input
        h, w, in_channels = input.shape
        output = np.zeros((h // self.kernel_size, w // self.kernel_size, in_channels))
        for region, x, y in self.generate(input):
            output[x, y] = np.amax(region, axis = (0, 1))
        return output
    
    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.cache['prev_input'].shape)

        for region, i, j  in self.generate(self.cache['prev_input']):
            h, w, f = region.shape
            amax = np.amax(region, axis = (0, 1))

            for h2 in range(h):
                for w2 in range(w):
                    for f2 in range(f):
                        if region[h2, w2, f2] == amax[f2]:
                            i2, j2 = i * 2 + h2, j * 2 + w2
                            d_L_d_input[i2, j2, f2] = d_L_d_out[i, j, f2]
        
        return d_L_d_input