import cv2
import numpy as np 
import os
import glob as gb # Python下的文件操作模块
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

BATCH_SIZE = 2000
PATH = "H:/google_image/"

def get_data(path = PATH,is_type = 'train',batch_size = BATCH_SIZE):
    '''
    从文件夹中读取一定数量的图片
    '''
    complete_path = PATH + is_type + '_image/'
    # 读取某文件夹下的所有图片
    train_jpg_path = gb.glob(complete_path + "*.jpg")
    train_png_path = gb.glob(complete_path + "*.png")
    train_path = train_jpg_path + train_png_path
    train_images = []

    # 随机选择BATCH_SIZE个图片
    index = [int(len(train_path) * p) for p in np.random.random(BATCH_SIZE)]
    choosen_path = [train_path[i] for i in index]
    
    # 添加标签
    label_list = []
    label_dict = get_label("train")
    
    for path in choosen_path:
        img = cv2.imread(path)
        # 有些图片损坏了，无法进行resize或者imshow，但是可以使用imread
        try:
            img = cv2.resize(img,(400,400))
            img = img.flatten()
            train_images.append(img)
            img_name = path.split('\\')[-1]
            label = label_dict[img_name]
            label_list.append(label)
            # cv2.imshow('img',img)
        except Exception as e:
            print(path)
            print(e)
        # cv2.waitKey(1)
        
    return train_images,label_list

def get_label(is_type):
    '''
    返回文件名与label组成的字典，方便对照
    '''
    label_df = pd.read_csv("H:/learning_notes/project/kaggle/iMaterialist Challenge/" + is_type + "_data.csv")
    name = label_df['name']
    label = label_df['label']
    return dict(zip(name,label))

if __name__ == "__main__":
    x,y = get_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    
    acc = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            acc += 1
    print(acc/len(y_test))