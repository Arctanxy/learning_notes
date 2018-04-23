#https://github.com/Six-wars/google-landmark-recognition-challenge/blob/186ba00a243c22b43e9b9bd8d00ea7d470911991/train.py

import pandas as pd 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K 
from keras.callbacks import EarlyStopping,ModelCheckpoint

# 图片像素尺寸
img_width,img_height = 350,350
train_samples = 10000
validation_samples = 5000
epochs = 50
batch_size = 10

train_data_dir = "H:/google_image/train_image/"
validation_data_dir = "H:/google_image/validation_image/"

#判断图片格式，有可能是通道在前，或者通道在后
if K.image_data_format() == "channels_first":
    input_shape = (1,img_width,img_height)
else:
    input_shape = (img_width,img_height,1)

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(14951,activation='softmax'))

monitor = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=5,verbose=0,mode='auto')
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5',verbose=0,save_best_only=True)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

train_datagen = ImageDataGenerator(
    rescale=1.0/255,#重缩放因子
    shear_range=0.2,#剪切强度
    zoom_range=0.2,#随机缩放的幅度[lower,upper] = [1 - zoom_range,1 + zoom_range]
    horizontal_flip=True#进行随机水平翻转
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model.fit_generator(
    train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs = epochs,
    validation_steps = validation_samples // batch_size,
    callbacks = [monitor,checkpoint],
    validation_data = validation_generator,
)

model.load_weights('best_weights.hdf5')
model.save('grey_model.h5')

scoreSeg = model.evaluate_generator(validation_generator,800)
print('Accuracy = ',scoreSeg[1])
