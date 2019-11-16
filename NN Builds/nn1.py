# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:48:38 2018

@author: Ahmet
"""

#   XOR.py-A very simple neural network to do exclusive or.
import numpy as np
 
epochs = 10000                                  # Number of iterations
i, h1, o = 2, 3, 1

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])
 
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
                                                # weights on layer inputs
w1 = np.random.uniform(size=(i, h1))
w2 = np.random.uniform(size=(h1, o))
 
for i in range(epochs):
 
    H1 = sigmoid(np.dot(X, w1))                  # hidden layer results
    Z = sigmoid(np.dot(H1, w2))                  # output layer results
    E = Y - Z                                   # how much we missed (error)
    dZ = E * sigmoid_(Z)                        # delta Z
    dH1 = dZ.dot(w2.T) * sigmoid_(H1)             # delta H
    w2 +=  H1.T.dot(dZ)                          # update output layer weights
    w1 +=  X.T.dot(dH1)                          # update hidden layer weights
     
print(Z)                # what have we learnt?