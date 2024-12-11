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
from sklearn.svm import SVC

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
y = df.iloc[:, -1]  # output data là cột cuối

# Tách dữ liệu
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Train SVM
print('SVM')
model = SVC(C=100, kernel='rbf',gamma=0.001,probability=True)
print('Model is training..............')
model.fit(x_train, y_train)
print('Done!!!\n')

print('ID3')
model1 = DecisionTreeClassifier(criterion='entropy', splitter='best')
print('Model ID3 is training......')
model1.fit(x_train, y_train)
print('Done')

# Save the trained model
try:
    with open('D:\py\Đồ án TTNT\svm_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model SVM saved successfully.")
    with open('D:\py\Đồ án TTNT\ID3_model.pkl', 'wb') as file:
        pickle.dump(model1, file)
    print("Model ID3 saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Check if the file was created
if os.path.exists('svm_model.pkl') and os.path.exists('ID3_model.pkl'):
    print("The model file 'svm_model.pkl' and 'ID3_model.pkl' was created successfully.")
else:
    print("The model file 'svm_model.pkl' and 'ID3_model.pkl' was not created.")
