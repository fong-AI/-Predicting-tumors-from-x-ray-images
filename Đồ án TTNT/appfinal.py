import pandas as pd
import numpy as np
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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
Categories=['benign','malignant','normal']                
flat_data_arr=[]                                           
target_arr=[]                                               
datadir='D:\py\Đồ án TTNT\Dataset_BUSI_with_GT'

# load,xử lý dư liệu
for i in Categories:
   
    print(f'loading...  {i}...')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))          
        flat_data_arr.append(img_resized.flatten())        
         #append thì lại là đẩy xuống dưới
        target_arr.append(Categories.index(i))           
    print(f'Done!!!')


flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)                                  
df['Target']=target
x=df.iloc[:,:-1]                                            
y=df.iloc[:,-1]                                             


# Tách dữ liệu 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10,stratify=y)
x_trainCNN = np.reshape(x_train, (-1, 150, 150, 3))
x_testCNN = np.reshape(x_test, (-1, 150, 150, 3))
x_trainLSTM = np.reshape(x_train, (-1, 1, 150, 150, 3))
x_testLSTM = np.reshape(x_test, (-1, 1, 150, 150, 3))
y_traindl = to_categorical(y_train, 3)
y_testdl = to_categorical(y_test, 3)

def load_models():
    with open('D:\\py\\Đồ án TTNT\\svm_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('D:\\py\\Đồ án TTNT\\ID3_model.pkl', 'rb') as file:
        model1 = pickle.load(file)
    model2 = load_model('D:\\py\\Đồ án TTNT\\CNN_model.h5')
    model3 = load_model('D:\\py\\Đồ án TTNT\\LSTM_model.h5')    
    return model,model1,model2,model3

model,model1,model2,model3 = load_models()


def cxtt():   
    y_predSVM=model.predict(x_test)
    cxSVM = accuracy_score(y_predSVM,y_test)*100
    y_predID3=model1.predict(x_test)
    cxID3 = accuracy_score(y_predID3,y_test)*100
    x_testCNN = np.reshape(x_test, (-1,1, 150, 150, 3))
    y_testCNN = np.argmax(y_testdl, axis=1)
    y_predCNN = model2.predict(x_testCNN)
    y_pred_classesCNN = np.argmax(y_predCNN, axis=1)
    accuracyCNN = accuracy_score(y_testCNN, y_pred_classesCNN) * 100
    y_testLSTM = np.argmax(y_testdl, axis=1)
    x_testLSTM = np.reshape(x_test, (-1,1, 150, 150, 3))
    y_predLSTM = model3.predict(x_testLSTM)
    y_pred_classesLSTM = np.argmax(y_predLSTM, axis=1)
    accuracyLSTM = accuracy_score(y_testLSTM, y_pred_classesLSTM) * 100
    # Hiển thị kết quả trong Label
    results = f"Độ chính xác SVM: {cxSVM:.2f}%\n"
    results += f"Độ chính xác ID3: {cxID3:.2f}%\n"
    results += f"Độ chính xác CNN: {accuracyCNN:.2f}%\n"
    results += f"Độ chính xác LSTM: {accuracyLSTM:.2f}%\n"
    
    # Cập nhật nội dung của Label
    result_label.config(text=results)
#Giao dien
windown=Tk()
windown.title("Image Classifier")
windown.geometry("550x300")


def openfn():
    filename = filedialog.askopenfilename()
    return filename
def open_img():
    anh = openfn()
    img = Image.open(anh)
    img = img.resize((150, 50), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel = Label(windown, image=img)
    panel.image = img
    panel.pack()
    img=imread(anh)
    plt.imshow(img)
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    kqSVM = Categories[model.predict(l)[0]] 
    kqID3 = Categories[model1.predict(l)[0]]
    img_resizedCNN = resize(img, (1,150, 150, 3))
    img_resizedCNN = np.expand_dims(img_resizedCNN, axis=0)  
    predCNN = model2.predict(img_resizedCNN)[0]
    index_predCNN = np.argmax(predCNN)
    kqCNN = Categories[index_predCNN]
    img = imread(anh)
    img_resizedLSTM = resize(img, (1,150, 150, 3))
    img_resizedLSTM = np.expand_dims(img_resizedLSTM, axis=0)  
    predLSTM = model3.predict(img_resizedLSTM)[0]
    index_predLSTM = np.argmax(predLSTM)
    kqLSTM = Categories[index_predLSTM]
    kqtt = f"Dự đoán SVM: {kqSVM}\n"
    kqtt += f"Dự đoán ID3: {kqID3}\n"
    kqtt += f"Dự đoán CNN: {kqCNN}\n"
    kqtt += f"Dự đoán LSTM: {kqLSTM}\n"
    kq_label.config(text=kqtt)
def clear():                                            
    windown.quit()            
    
btn = Button(windown, text='open image', command=open_img).pack(side = TOP, fill = BOTH)
cx = Button(windown, text='Độ chính xác thuật toán', command=cxtt).pack(side = TOP, fill = BOTH)
clearScr = Button(windown, text='Đóng', command=clear).pack(side = TOP,  fill = BOTH)
result_label = Label(windown, text="", justify=LEFT)
result_label.pack()
kq_label = Label(windown, text="", justify=LEFT)
kq_label.pack()
windown.mainloop()
