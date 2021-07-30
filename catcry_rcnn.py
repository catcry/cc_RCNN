# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 00:02:57 2021

@author: catcry
"""
#%% imports
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#cv2 or opencv library for selective search



#%%             Intersection over Union Calculation
#
def iou (bbox_groud, bbox_proposal):
    
    if  bbox_groud['xl'] > bbox_groud['xr'] or\
        bbox_groud['yt'] > bbox_groud['yb'] or\
        bbox_proposal['xl'] > bbox_proposal['xr'] or\
        bbox_proposal['yt'] > bbox_proposal['yb'] :
            print ('invalid boundaries')
            
        
    xl_intersection = max(bbox_groud['xl'],bbox_proposal['xl'])
    xr_intersection = min(bbox_groud['xr'],bbox_proposal['xr'])
    yt_intersection = max(bbox_groud['yt'],bbox_proposal['yt'])
    yb_intersection = min(bbox_groud['yb'],bbox_proposal['yb'])
    if xl_intersection > xr_intersection or\
       yt_intersection > yb_intersection:
       return 0.0
    intersection = (xr_intersection - xl_intersection) * (yb_intersection - yt_intersection)
   
    bbox_groud_area = (bbox_groud['xr'] - bbox_groud['xl']) * (bbox_groud['yb'] - bbox_groud['yt'])
    bbox_proposal_area = (bbox_proposal['xr'] - bbox_proposal['xl']) * (bbox_proposal['yb'] - bbox_proposal['yt'])
    
    union =  bbox_groud_area + bbox_proposal_area - intersection
    iou_o = intersection/union
    
    assert iou_o >= 0
    assert iou_o <= 1
    
    return iou_o

#%%

# img_path = r"f:\git\mask\RCNN\Images"
# bbox_gt_path = r"f:\git\mask\RCNN\Airplanes_Annotations"


sel_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
img_path = "/home/catcry/catcry_RCNN/Images"
bbox_gt_path = "/home/catcry/catcry_RCNN/Airplanes_Annotations"

train_images=[]
train_labels=[]

for e,i in enumerate(os.listdir(bbox_gt_path)):
    try:
        if i.startswith("airplane"):
            filename = i.split(".")[0]+".jpg"
            print(e,filename)
            image = cv2.imread(os.path.join(img_path,filename))
            df = pd.read_csv(os.path.join(bbox_gt_path,i))
            gtvalues=[]
            for row in df.iterrows():
                xl = int(row[1][0].split(" ")[0])
                yt = int(row[1][0].split(" ")[1])
                xr = int(row[1][0].split(" ")[2])
                yb = int(row[1][0].split(" ")[3])
                gtvalues.append({"xl":xl,"xr":xr,"yt":yt,"yb":yb})
            sel_search.setBaseImage(image)
            sel_search.switchToSelectiveSearchFast()
            ssresults = sel_search.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e,result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x,y,w,h = result
                        iou_o = iou(gtval,{"xl":x,"xr":x+w,"yt":y,"yb":y+h})
                        print (iou_o)
                        if counter < 30:
                            if iou_o > 0.70:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter += 1
                        else :
                            fflag =1
                        if falsecounter <30:
                            if iou_o < 0.3:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        print("error in "+filename)
        continue
    
train_imgs = np.array(train_images)
train_lbls = np.array(train_labels)


    
#%%     VGG-16 Model with pretrained weights

vggmodel = VGG16(weights='imagenet', include_top=True) 

for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

# Extract the -2 layer as feature map to dense it into two categories:
f_map = vggmodel.layers[-2].output
air_plane = Dense(2, activation="softmax")(f_map)
model = Model(vggmodel.input, air_plane)
opt = Adam(lr=0.0001)
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model.summary()



#%%       Generation of Train and validation sets 
train_lbl_1hot_lst=[]
for record in train_lbls:
    train_lbl_1hot_lst.append([record,1-record])
train_lbls_1hot = np.array(train_lbl_1hot_lst)
    
Xtrain, Xvalid , Ytrain, Yvalid = train_test_split(train_imgs,train_lbls_1hot,test_size=0.10)
XYtrain_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
XYtrain_flow = XYtrain_gen.flow(x=Xtrain, y=Ytrain)
XYvalid_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
XYvalid_flow = XYvalid_gen.flow(x=Xvalid, y=Yvalid)




#%%

checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
history = model.fit_generator(generator= XYtrain_flow, steps_per_epoch= 10, epochs= 100, validation_data=XYvalid_flow, validation_steps=2, callbacks=[checkpoint,early])


#%%             Testing the trained model

for e,file in enumerate(os.listdir(img_path)):
    if file.startswith("4"):
        img = cv2.imread(os.path.join(img_path,file))
        sel_search.setBaseImage(img)
        sel_search.switchToSelectiveSearchFast()
        ssresults = sel_search.process()
        imout = img.copy()
        for e,result in enumerate(ssresults):
            if e < 2000:
                x,y,w,h = result
                timage = imout[y:y+h,x:x+w]
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out= model.predict(img)
                if out[0][0] > 0.70:
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(imout, 'airplane', (x-1,y-1),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
        plt.figure()
        plt.imshow(imout)
        break
import time

a = time.time()    
img = cv2.imread(os.path.join(img_path,file))
sel_search.setBaseImage(img)
sel_search.switchToSelectiveSearchFast()
ssresults = sel_search.process()
imout = img.copy()
for e,result in enumerate(ssresults):
    if e < 2000:
        x,y,w,h = result
        timage = imout[y:y+h,x:x+w]
        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)
        out= model.predict(img)
        if out[0][0] > 0.70:
            cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(imout, 'airplane', (x-1,y-1),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
plt.figure()
plt.imshow(imout)   
b = time.time() - a
print (b)