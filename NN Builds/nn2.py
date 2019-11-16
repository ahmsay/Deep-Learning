# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 17:01:28 2018

@author: Ahmet
"""

import numpy as np

x = np.array([[0,0], [0,0.5], [0,1], [0.5,0], [0.5,0.5], [0.5,1], [1,0], [1,0.5], [1,1]])
y = np.array([[1], [0], [0], [0], [1], [0], [0], [0], [1]])

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(3, activation = "tanh", input_dim = 2))
classifier.add(Dense(3, activation = "tanh"))
classifier.add(Dense(1, activation = "tanh"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#classifier.fit(X, Y, epochs = 10000)
y_pred = classifier.predict(x)
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, y_pred)
print(cm)
"""