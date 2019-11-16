import numpy as np
import os
from PIL import Image
import glob
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

samples_needed = 300
image_size = (64, 64)
num_of_classes = 9
test_ratio = 0.30
input_shape = image_size + (3,)
batch_size = 128
epochs = 150

directories = []
for index, (root, dirs, files) in zip(range(1), os.walk('./dataset')):
    directories = dirs

image_list = []
class_list = []

for directory in directories:
    for index, filename in zip(range(samples_needed), glob.glob('./dataset/' + directory + '/*.jpg')):
        im = Image.open(filename)
        im = im.resize(image_size, Image.ANTIALIAS)
        im = image.img_to_array(im)
        im = im / 255
        image_list.append(im)
        class_list.append(directory)

x = np.asarray(image_list)

le = LabelEncoder()
y = le.fit_transform(class_list)
y = keras.utils.to_categorical(y, num_of_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_ratio, random_state = 0)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])