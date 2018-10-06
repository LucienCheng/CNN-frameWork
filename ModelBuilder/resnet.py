"""
@Author:clfight
@Date:18-10-6
@Desc:

"""
# import the necessary packages
from keras.models import Sequential,Model
from keras import backend as K
from keras.layers import BatchNormalization ,AveragePooling2D, Input, Flatten,add,MaxPooling2D,Activation,Dense,Conv2D
from keras.regularizers import l2

class ResnetModel:
    @staticmethod
    def resnet_block(inputs, num_filters=64,
                     kernel_size=(3, 3), strides=1,
                     activation='relu'):
        x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        if (activation):
            x = Activation('relu')(x)
        return x


    # 建一个20层的ResNet网络
    @staticmethod
    def resnet_v1(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            input_shape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        inputs = Input(shape=input_shape)  # Input层，用来当做占位使用

        # 第一层
        x = ResnetModel.resnet_block(inputs)
        print('layer1,xshape:', x.shape)
        # 第2~7层
        for i in range(6):
            a = ResnetModel.resnet_block(inputs=x)
            b = ResnetModel.resnet_block(inputs=a, activation=None)
            x = add([x, b])
            x = Activation('relu')(x)
        # out：32*32*16
        # 第8~13层
        for i in range(6):
            if i == 0:
                a = ResnetModel.resnet_block(inputs=x, strides=2, num_filters=64)
            else:
                a = ResnetModel.resnet_block(inputs=x, num_filters=64)
            b = ResnetModel.resnet_block(inputs=a, activation=None, num_filters=64)
            if i == 0:
                x = Conv2D(64, kernel_size=(3, 3), strides=2, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
            x = add([x, b])
            x = Activation('relu')(x)
        # out:16*16*32
        # 第14~19层
        for i in range(6):
            if i == 0:
                a = ResnetModel.resnet_block(inputs=x, strides=2, num_filters=128)
            else:
                a = ResnetModel.resnet_block(inputs=x, num_filters=128)

            b = ResnetModel.resnet_block(inputs=a, activation=None, num_filters=128)
            if i == 0:
                x = Conv2D(128, kernel_size=(3, 3), strides=2, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
            x = add([x, b])  # 相加操作，要求x、b shape完全一致
            x = Activation('relu')(x)
        # out:8*8*64
        # 第20层
        x = AveragePooling2D(pool_size=2)(x)
        # out:4*4*64
        y = Flatten()(x)
        # out:1024
        outputs = Dense(classes, activation='softmax',
                        kernel_initializer='he_normal')(y)

        # 初始化模型
        # 之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
        model = Model(inputs=inputs, outputs=outputs)
        return model