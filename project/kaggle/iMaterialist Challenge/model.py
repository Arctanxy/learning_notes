import cv2
import numpy as np 
import os
import glob as gb # Python下的文件操作模块

BATCH_SIZE = 100
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
    count = 0
    print("^^Reading Images^^")
    print(train_path)
    for path in train_path:
        print(count)
        if count == BATCH_SIZE - 1:
            break
        img = cv2.imread(path)
        # 有些图片损坏了，无法进行resize或者imshow，但是可以使用imread
        try:
            img = cv2.resize(img,(400,400))
            train_images.append(img)
            count += 1
            # cv2.imshow('img',img)
        except Exception as e:
            print(path)
            print(e)
        # cv2.waitKey(1)
    
    return train_images

def get_label(path,is_type):
    label_df = 

if __name__ == "__main__":
    print(train_data())
