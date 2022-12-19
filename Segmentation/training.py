
import segmentation_models as sm
import numpy as np
import tensorflow as tf
import cv2
import glob
import os
from tensorflow import keras

# Give the path of ground truth and mask images folder
a = input("enter the path of ground truth images")
b = input("enter the path of masked images")

# preprocessing of images
train_image = []
for directory_path in glob.glob(a):
    for img_path in glob.glob(os.path.join(directory_path,"*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        res  = cv2.resize(img, (512,512))
        train_image.append(res)
train_images = np.array(train_image)

train_mask = []
for directory_path in glob.glob(b):
    for mask_path in glob.glob(os.path.join(directory_path,"*.tif")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        maskres  = cv2.resize(mask, (512,512))
        train_mask.append(maskres)
train_mask = np.array(train_mask)

x_train = train_images
y_train = train_mask
y_train = np.expand_dims(y_train,axis=3)

print(x_train.shape)
print(y_train.shape)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, encoder_weights=None)

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

model.fit(
   x=x_train,
   y=y_train,
   batch_size=8,
   epochs=12,

)

model.save("roadseg_epoch12_dataset867.h5")


 


