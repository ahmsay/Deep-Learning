import numpy as np

class Layer:
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        
    def sigmoid (self, x): return 1/(1 + np.exp(-x))
    def sigmoid_(self, x): return x * (1 - x)
    
    def relu (self, x): return (x > 0) * x
    def relu_(self, x): return (x > 0) * 1
    
    def func(self, x):
        if(self.activation == "sigmoid"):
            return self.sigmoid(x)
        elif(self.activation == "relu"):
            return self.relu(x)
        
    def derv(self, x):
        if(self.activation == "sigmoid"):
            return self.sigmoid_(x)
        elif(self.activation == "relu"):
            return self.relu_(x)