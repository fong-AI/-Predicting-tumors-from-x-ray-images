import pandas as pd
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

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

# Định nghĩa tham số cần tìm kiếm
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
}

# Khởi tạo mô hình Decision Tree Classifier
dt = DecisionTreeClassifier()

# Sử dụng GridSearchCV
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# In ra tham số tốt nhất
print(f"Best parameters found: {grid_search.best_params_}")

# Đào tạo mô hình với tham số tốt nhất
print(grid_search.best_estimator_)

# Dự đoán và đánh giá mô hình
y_pred = grid_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Độ chính xác của mô hình: {accuracy}%")
