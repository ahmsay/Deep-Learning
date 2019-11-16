from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target
y = y.reshape((150,1))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 0)

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork(4,8,6,1)
nn.update(x_train, y_train, 10000, 0.2)
y_pred = nn.predict(x_test)

for i in range(y_pred.shape[0]):
    if y_pred[i] < 0.33:
        y_pred[i] = 0
    elif y_pred[i] >= 0.33 and y_pred[i] < 0.67:
        y_pred[i] = 0.5
    else:
        y_pred[i] = 1
        
print("Testing incorrect predictions:", (y_test != y_pred).sum(), "/", y_test.shape[0])

y_pred = nn.predict(x_train)

for i in range(y_pred.shape[0]):
    if y_pred[i] < 0.33:
        y_pred[i] = 0
    elif y_pred[i] >= 0.33 and y_pred[i] < 0.67:
        y_pred[i] = 0.5
    else:
        y_pred[i] = 1
        
print("Training incorrect predictions:", (y_train != y_pred).sum(), "/", y_train.shape[0])