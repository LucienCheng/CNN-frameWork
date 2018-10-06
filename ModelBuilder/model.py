"""
@Author:clfight
@Date:18-10-6
@Desc:

"""
# import the necessary packages
# dropout:有一定比例的点失效
# Flatten：将多维张量变成一维，为了全连接准备
# MaxPolling2D:最大化池层
# Activation：激活函数
# Dense：全连接层
# Conv2D：卷积
from keras.models import Sequential,Model
from keras import backend as K
from keras.layers.core import Dropout
from keras.layers import Flatten,MaxPooling2D,Activation,Dense,Conv2D




class Model:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecturel
        return model
    # ResNet Block
