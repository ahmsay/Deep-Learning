import numpy as np

class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = layers
        self.setWeights()
        
    def sigmoid (self, x): return 1/(1 + np.exp(-x))
    def sigmoid_(self, x): return x * (1 - x)
        
    def setWeights(self):
        self.weights = []
        self.hiddens = []
        self.dHiddens = []
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.uniform(size=(self.layers[i], self.layers[i+1])))
            self.hiddens.append(0)
            self.dHiddens.append(0)
        self.length = len(self.hiddens) - 1
            
    def update(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
                self.hiddens[0] = self.sigmoid(np.dot(x_train, self.weights[0]))
                for i in range(self.length):
                    self.hiddens[i+1] = self.sigmoid(np.dot(self.hiddens[i], self.weights[i+1]))
                E = y_train - self.hiddens[self.length]
                self.dHiddens[self.length] = E * self.sigmoid_(self.hiddens[self.length])
                for i in reversed(range(self.length)):
                    self.dHiddens[i] = self.dHiddens[i+1].dot(self.weights[i+1].T) * self.sigmoid_(self.hiddens[i])
                for i in reversed(range(self.length)):
                    self.weights[i+1] += learning_rate * self.hiddens[i].T.dot(self.dHiddens[i+1])
                self.weights[0] += learning_rate * x_train.T.dot(self.dHiddens[0])
        return self.hiddens[self.length]
        
    def predict(self, x_test):
        self.hiddens[0] = self.sigmoid(np.dot(x_test, self.weights[0]))
        for i in range(self.length):
            self.hiddens[i+1] = self.sigmoid(np.dot(self.hiddens[i], self.weights[i+1]))
        return self.hiddens[self.length]