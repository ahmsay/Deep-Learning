import numpy as np
from Layer import Layer

class NewNetwork:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def sigmoid (self, x): return 1/(1 + np.exp(-x))
    def sigmoid_(self, x): return x * (1 - x)
        
    def setWeights(self):
        self.weights = []
        self.hiddens = []
        self.dHiddens = []
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.uniform(size=(self.layers[i].neurons, self.layers[i+1].neurons)))
            self.hiddens.append(0)
            self.dHiddens.append(0)
        self.length = len(self.hiddens) - 1
        
    def train(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
                self.hiddens[0] = self.layers[1].func(np.dot(x_train, self.weights[0]))
                for i in range(self.length):
                    self.hiddens[i+1] = self.layers[i+1].func(np.dot(self.hiddens[i], self.weights[i+1]))
                E = y_train - self.hiddens[self.length]
                self.dHiddens[self.length] = E * self.layers[self.length].derv(self.hiddens[self.length])
                for i in reversed(range(self.length)):
                    self.dHiddens[i] = self.dHiddens[i+1].dot(self.weights[i+1].T) * self.layers[self.length].derv(self.hiddens[i])
                for i in reversed(range(self.length)):
                    self.weights[i+1] += learning_rate * self.hiddens[i].T.dot(self.dHiddens[i+1])
                self.weights[0] += learning_rate * x_train.T.dot(self.dHiddens[0])
        return self.hiddens[self.length]
    
    def predict(self, x_test):
        self.hiddens[0] = self.layers[1].func(np.dot(x_test, self.weights[0]))
        for i in range(self.length):
            self.hiddens[i+1] = self.layers[i+1].func(np.dot(self.hiddens[i], self.weights[i+1]))
        return self.hiddens[self.length]
    
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])

nn = NewNetwork()
nn.add(Layer(2, ""))
nn.add(Layer(3, "sigmoid"))
nn.add(Layer(3, "sigmoid"))
nn.add(Layer(1, "sigmoid"))
nn.setWeights()

nn.train(X, Y, 5000, 1.0)
pred = nn.predict(X)