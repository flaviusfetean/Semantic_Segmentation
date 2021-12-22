from tensorflow.keras.models import load_model
from utils import convert_one_hot_to_rgb
from imutils import paths
import numpy as np
import cv2

palette = [  # in rgb
            [0, 0, 0],  # 0: None
            [70, 70, 70],  # 1: Buildings
            [190, 153, 153],  # 2: Fences
            [192, 192, 192],  # 3: Other
            [220, 20, 60],  # 4: Pedestrians
            [153, 153, 153],  # 5: Poles
            [0, 255, 0],  # 6: RoadLines
            [128, 64, 128],  # 7: Roads
            [244, 35, 232],  # 8: Sidewalks
            [107, 142, 35],  # 9: Vegetation
            [0, 0, 142],  # 10: Vehicles
            [102, 102, 156],  # 11: Walls
            [220, 220, 0]]  # 11: Traffic signs

images_paths = list(paths.list_images("dataset_test"))
images = [cv2.imread(img_path) for img_path in images_paths]

images = [cv2.resize(img, (160, 160)) for img in images]
images_preprocessed = np.array(images, dtype='float') / 255.0

model = load_model("model")

print(model.summary())

preds = model.predict(images_preprocessed, batch_size=8)
conv_preds = [convert_one_hot_to_rgb(pred, palette) for pred in preds]

collage = []
for image, pred in zip(images, conv_preds):
    seg_pred = pred[0]
    coll = np.hstack([image, seg_pred])
    collage.append(coll)

for coll in collage:
    cv2.imshow("Semantic Segmentation", coll)
    cv2.waitKey(0)
