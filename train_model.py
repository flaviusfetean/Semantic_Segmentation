from dense_unet import define_model
from sklearn.model_selection import train_test_split
from utils import one_hot_encoding
import cv2
import os
import numpy as np
from imutils import paths

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
img_shape = (160, 160, 3)
num_classes = 13

print("[INFO] Building model...")

model = define_model(img_shape, num_classes, compression_factor=0.3, num_groups=2)

print("[INFO] Collecting dataset...")

Xs, Ys = list(paths.list_images("dataset/rgb")), list(paths.list_images("dataset/seg_raw"))

print("[INFO] Preprocessing dataset")

Xs = [cv2.resize(cv2.imread(x), img_shape[:2]) for x in Xs]
Ys = [cv2.resize(cv2.imread(y), img_shape[:2]) for y in Ys]

trainX, testX, trainY, testY = train_test_split(Xs, Ys, test_size=0.2, random_state=42)

trainX = np.array(trainX, dtype='float') / 255.0
testX = np.array(testX, dtype='float') / 255.0

trainY = one_hot_encoding(trainY, num_classes)
testY = one_hot_encoding(testY, num_classes)

print("[INFO] Training model...")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.fit(trainX, trainY, verbose=1, batch_size=8, epochs=20, validation_data=(testX, testY))

print("[INFO] Training ended successfully")

print("[INFO] Saving model")
model.save("moodel")