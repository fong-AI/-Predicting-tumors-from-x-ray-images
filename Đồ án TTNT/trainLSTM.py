# import pandas as pd
# import os
# from skimage.transform import resize
# from skimage.io import imread
# import numpy as np
# import matplotlib.pyplot as plt
# from tkinter import *
# import tkinter
# from sklearn.model_selection import train_test_split
# from tkinter import filedialog
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from tkinter import messagebox
# from PIL import ImageTk, Image
# from tkinter import filedialog, ttk
# from sklearn.tree import DecisionTreeClassifier
# import pickle
# from keras.models import Sequential
# from tensorflow.keras import models, Model
# from tensorflow.keras.layers import LSTM, Input, Dense, Dropout,Flatten
# # from keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical

# Categories = ['benign', 'malignant', 'normal']
# flat_data_arr = []  # khởi tạo input array
# target_arr = []  # khởi tạo output array
# datadir = 'D:\\pyyy\\Đồ án TTNT\\Dataset_BUSI_with_GT'

# # load, xử lý dư liệu
# for i in Categories:
#     print(f'loading...  {i}...')
#     path = os.path.join(datadir, i)
#     for img in os.listdir(path):
#         img_array = imread(os.path.join(path, img))
#         img_resized = resize(img_array, ( 150, 3))  # thay đổi kích thước ảnh
#         flat_data_arr.append(img_resized.flatten())  # flatten làm phẳng bức ảnh đa chiều thành 1 chiều tạp ra 1 dãy dữ liệu
#         target_arr.append(Categories.index(i))
#     print(f'Done!!!')

# flat_data = np.array(flat_data_arr)
# target = np.array(target_arr)

# df = pd.DataFrame(flat_data)  # dataframe
# df['Target'] = target
# x = df.iloc[:, :-1]  # input data lấy đến kế cuối
# y = df.iloc[:, -1]  # output data là dòng cuối

# # Tách dữ liệu
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=130, stratify=y)
# x_train = np.array(x_train).reshape(x_train.shape[0], 150,3)
# x_test = np.array(x_test).reshape(x_test.shape[0],  150,3)
# # x_train = np.reshape(x_train, ( 150, 150, 3))
# # x_test = np.reshape(x_test, ( 150, 150, 3))
# # y_train = to_categorical(y_train, 3)
# # y_test = to_categorical(y_test, 3)
# model3 = Sequential()
# model3.add(LSTM(units=128, return_sequences=True, input_shape=( 150 ,3)))
# model3.add(LSTM(units=64))

# # hidden layer
# model3.add(Dense(128, activation='relu'))
# # output layer
# model3.add(Dense(3, activation='softmax'))

# # compiling the sequential model
# model3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # training the model for 10 epochs
# model3.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# # model3= Sequential()
# # Save the trained model
# try:
#     with open('D:\pyyy\Đồ án TTNT\LSTM_model.pkl', 'wb') as file:
#         pickle.dump(model3, file)
#     print("Model LSTM saved successfully.")
# except Exception as e:
#     print(f"Error saving model: {e}")

# # Check if the file was created
# if os.path.exists('LSTM_model.pkl'):
#     print("The model file 'LSTM_model.pkl' was created successfully.")
# else:
#     print("The model file 'LSTM_model.pkl' was not created.")


# import pandas as pd
# import os
# from skimage.transform import resize
# from skimage.io import imread
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.utils import to_categorical
# import pickle

# Categories = ['benign', 'malignant', 'normal']
# flat_data_arr = []  # Initialize input array
# target_arr = []  # Initialize output array
# datadir = 'D:\\pyyy\\Đồ án TTNT\\Dataset_BUSI_with_GT'

# # Load and process data
# for i in Categories:
#     print(f'loading...  {i}...')
#     path = os.path.join(datadir, i)
#     for img in os.listdir(path):
#         img_array = imread(os.path.join(path, img))
#         img_resized = resize(img_array, (150, 150, 3))  # Resize images to (150, 150, 3)
#         flat_data_arr.append(img_resized)
#         target_arr.append(Categories.index(i))
#     print(f'Done!!!')

# flat_data = np.array(flat_data_arr)
# target = np.array(target_arr)

# x = flat_data  # Input data
# y = target  # Output data

# # One-hot encode the target labels
# y = to_categorical(y, num_classes=3)

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=130, stratify=y)

# # Define the LSTM model
# model3 = Sequential()
# model3.add(LSTM(units=128, return_sequences=True, input_shape=(150, 150, 3)))
# model3.add(LSTM(units=64))

# # Hidden layer
# model3.add(Dense(128, activation='relu'))
# # Output layer
# model3.add(Dense(3, activation='softmax'))

# # Compile the sequential model
# model3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # Train the model for 10 epochs
# model3.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# # Save the trained model
# model3.save('D:\\pyyy\\Đồ án TTNT\\LSTM_model.h5')
# print("Model LSTM saved successfully.")

# # Check if the file was created
# if os.path.exists('D:\\pyyy\\Đồ án TTNT\\LSTM_model.h5'):
#     print("The model file 'LSTM_model.h5' was created successfully.")
# else:
#     print("The model file 'LSTM_model.h5' was not created.")



# import pandas as pd
# import os
# from skimage.transform import resize
# from skimage.io import imread
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv2D, Flatten, MaxPooling2D,Reshape, TimeDistributed
# from tensorflow.keras.utils import to_categorical
# import pickle

# Categories = ['benign', 'malignant', 'normal']
# flat_data_arr = []  # khởi tạo input array
# target_arr = []  # khởi tạo output array
# datadir = 'D:\\pyyy\\Đồ án TTNT\\Dataset_BUSI_with_GT'

# # load, xử lý dư liệu
# for i in Categories:
#     print(f'loading...  {i}...')
#     path = os.path.join(datadir, i)
#     for img in os.listdir(path):
#         img_array = imread(os.path.join(path, img))
#         img_resized = resize(img_array, (150, 150, 3))  # thay đổi kích thước ảnh
#         flat_data_arr.append(img_resized.flatten())  # flatten làm phẳng bức ảnh đa chiều thành 1 chiều tạp ra 1 dãy dữ liệu
#         target_arr.append(Categories.index(i))
#     print(f'Done!!!')

# flat_data = np.array(flat_data_arr)
# target = np.array(target_arr)

# df = pd.DataFrame(flat_data)  # dataframe
# df['Target'] = target
# x = df.iloc[:, :-1]  # input data lấy đến kế cuối
# y = df.iloc[:, -1]  # output data là dòng cuối

# # Tách dữ liệu
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=130, stratify=y)
# x_train = np.reshape(x_train, (-1, 150, 150, 3))
# x_test = np.reshape(x_test, (-1, 150, 150, 3))
# y_train = to_categorical(y_train, 3)
# y_test = to_categorical(y_test, 3)
# model = Sequential()#
# model.add(TimeDistributed(Conv2D(64,(3,3),activation='relu'),input_shape=(150,150,3)))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

# model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu')))

# model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

# model.add(TimeDistributed(Flatten()))

# #RNN
# model.add(LSTM(100,return_sequences=False))

# model.add(Dense(2,activation='sigmoid'))#


# # Compile the sequential model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # Train the model for 10 epochs
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# # Save the trained model
# model.save('D:\\pyyy\\Đồ án TTNT\\LSTM_model.h5')
# print("Model LSTM saved successfully.")

# # Check if the file was created
# if os.path.exists('D:\\pyyy\\Đồ án TTNT\\LSTM_model.h5'):
#     print("The model file 'LSTM_model.h5' was created successfully.")
# else:
#     print("The model file 'LSTM_model.h5' was not created.")

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten,Dropout
from tensorflow.keras.utils import to_categorical
import pickle

Categories = ['benign', 'malignant', 'normal']
flat_data_arr = []  # khởi tạo input array
target_arr = []  # khởi tạo output array
datadir = 'D:\py\Đồ án TTNT\Dataset_BUSI_with_GT'

# Load và xử lý dữ liệu
for i in Categories:
    print(f'loading...  {i}...')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))  # thay đổi kích thước ảnh
        flat_data_arr.append(img_resized)  # Lưu ảnh đã thay đổi kích thước vào mảng
        target_arr.append(Categories.index(i))
    print(f'Done!!!')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Tách dữ liệu
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.20, random_state=20)
x_train = np.reshape(x_train, (-1, 1, 150, 150, 3))
x_test = np.reshape(x_test, (-1, 1, 150, 150, 3))
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# Định nghĩa mô hình
model = Sequential()

# Các lớp CNN với TimeDistributed
# model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'), input_shape=( 1,150, 150, 3)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
# Lớp LSTM
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.1))
# Lớp fully connected
model.add(Dense(128, activation='relu'))

# Lớp đầu ra
model.add(Dense(3, activation='softmax'))

# Compile the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model for 10 epochs
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save('D:\\py\\Đồ án TTNT\\LSTM_model.h5')
# Save the trained model
# try:
#     with open('D:\py\Đồ án TTNT\LSTM_model.pkl', 'wb') as file:
#         pickle.dump(model, file)
#     print("Model LSTM saved successfully.")
# except Exception as e:
#     print(f"Error saving model: {e}")

# Check if the file was created
if os.path.exists('D:\\py\\Đồ án TTNT\\LSTM_model.h5'):
    print("The model file 'LSTM_model.h5' was created successfully.")
else:
    print("The model file 'LSTM_model.h5' was not created.")
