"""
@Author:clfight
@Date:18-10-6
@Desc:

"""
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os
import logging
"""
test_rate:float,表示的是选取图片的比例
norm_size：int,图片大小
label_dict：dict,分类字典
CLASS_NUM：分类个数
"""

def load_data(path,test_rate,norm_size,label_dict,CLASS_NUM):
    print("[INFO] loading images...")
    test_data = []
    test_labels = []
    train_data = []
    train_labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    test_num = int(test_rate * len(imagePaths))
    logging.info("total image num is %s", len(imagePaths))
    logging.info("test data num is %s", test_num)
    count = 0
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        logging.debug("image path is %s", imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        # data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label_name = imagePath.split(os.path.sep)[-1].split(".")[0].split("2018")[0]
        logging.debug("label name is %s", label_name)
        try:
            label = int(label_dict[label_name])
        except:
            label = 11
        logging.debug("label is %s", label)
        if (count < test_num):
            test_data.append(image)
            test_labels.append(label)
        else:
            train_data.append(image)
            train_labels.append(label)
        count += 1;

    # scale the raw pixel intensities to the range [0, 1]
    test_data = np.array(test_data, dtype="float") / 255.0
    train_data = np.array(train_data, dtype="float") / 255.0
    test_labels = np.array(test_labels)
    trian_labels = np.array(train_labels)

    # convert the labels from integers to vectors
    test_labels = to_categorical(test_labels, num_classes=CLASS_NUM)
    train_labels = to_categorical(train_labels, num_classes=CLASS_NUM)
    return test_data, test_labels, train_data, train_labels