#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")

X = data.iloc[:,3:13].values
Y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(6, init = "uniform", activation = "relu", input_dim = 12))
classifier.add(Dense(6, init = "uniform", activation = "relu"))
classifier.add(Dense(1, init = "uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs = 50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)