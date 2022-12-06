import numpy as np
from ConvLayer import ConvLayer
from MaxPoolLayer import MaxPool
from SoftMaxLayer import SoftMax
from ReluLayer import Relu

class Model:

    def __init__(self):
        self.conv = ConvLayer(1, 8, 3)
        self.maxpool = MaxPool(2)
        self.relu = Relu()
        self.softmax = SoftMax(13 * 13 * 8, 10)
    
    def forward(self, image, label):
        out = self.conv.forward((image / 255) - 0.5)
        out = self.maxpool.forward(out)
        out = self.softmax.forward(out)
    
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc
    
    def train(self, im, label, rate):
        out, loss, acc = self.forward(im, label)

        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        gradient = self.softmax.backprop(gradient, rate)
        gradient = self.maxpool.backprop(gradient)
        gradient = self.conv.backprop(gradient, rate)
        return loss, acc
