# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:15:07 2018

@author: Ahmet
"""

import numpy as np

epochs = 10000                
i, h1, h2, o = 2, 3, 3, 1

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])

def sigmoid (x): return 1/(1 + np.exp(-x))
def sigmoid_(x): return x * (1 - x)

w1 = np.random.uniform(size=(i, h1))
w2 = np.random.uniform(size=(h1, h2))
w3 = np.random.uniform(size=(h2, o))

for i in range(epochs):
    H1 = sigmoid(np.dot(X, w1))
    H2 = sigmoid(np.dot(H1, w2))
    Z = sigmoid(np.dot(H2, w3))
    E = Y - Z
    dZ = E * sigmoid_(Z)
    dH2 = dZ.dot(w3.T) * sigmoid_(H2)
    dH1 = dH2.dot(w2.T) * sigmoid_(H1)
    w3 += H2.T.dot(dZ)
    w2 += H1.T.dot(dH2)
    w1 += X.T.dot(dH1)
print(Z)