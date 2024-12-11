import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from sklearn import svm
from sklearn.metrics import accuracy_score
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog, ttk
from sklearn.tree import DecisionTreeClassifier
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D,TimeDistributed
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

Categories = ['benign', 'malignant', 'normal']
flat_data_arr = []  # khởi tạo input array
target_arr = []  # khởi tạo output array
datadir = 'D:\py\Đồ án TTNT\Dataset_BUSI_with_GT'

# load, xử lý dư liệu
for i in Categories:
    print(f'loading...  {i}...')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))  # thay đổi kích thước ảnh
        flat_data_arr.append(img_resized.flatten())  # flatten làm phẳng bức ảnh đa chiều thành 1 chiều tạp ra 1 dãy dữ liệu
        target_arr.append(Categories.index(i))
    print(f'Done!!!')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
x = df.iloc[:, :-1]  # input data lấy đến kế cuối
y = df.iloc[:, -1]  # output data là dòng cuối

# Tách dữ liệu
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=20)
x_train = np.reshape(x_train, (-1,1, 150, 150, 3))
x_test = np.reshape(x_test, (-1,1, 150, 150, 3))
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
# building a linear stack of layers with the sequential model
model2 = Sequential()
# convolutional layer
# model2.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# #model2.add(MaxPool2D(pool_size=(1,1)))
# model2.add(MaxPooling2D((2, 2)))
# model2.add(Conv2D(32, (3, 3), activation='relu'))
# model2.add(MaxPooling2D((2, 2)))
# model2.add(Conv2D(64, (3, 3), activation='relu'))
# model2.add(MaxPooling2D((2, 2)))
# model2.add(Dropout(0.3))
model2.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(1, 150, 150, 3)))
model2.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model2.add(TimeDistributed(Conv2D(54, (3, 3), activation='relu')))
model2.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model2.add(TimeDistributed(Flatten()))
# flatten output of conv
model2.add(Flatten())
# hidden layer5
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
# output layer
model2.add(Dense(3, activation='softmax'))

# compiling the sequential model
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model2.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model2.save('D:\\py\\Đồ án TTNT\\CNN_model.h5')
# model3= Sequential()
# Save the trained model
# try:
#     with open('D:\py\Đồ án TTNT\CNN_model.pkl', 'wb') as file:
#         pickle.dump(model2, file)
#     print("Model CNN saved successfully.")
# except Exception as e:
#     print(f"Error saving model: {e}")

# Check if the file was created
if os.path.exists('D:\\py\\Đồ án TTNT\\CNN_model.h5'):
    print("The model file 'CNN_model.pkl' was created successfully.")
else:
    print("The model file 'svm_model.pkl' was not created.")
