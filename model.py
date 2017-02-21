from keras.layers import merge, Lambda, Convolution2D, Deconvolution2D, MaxPooling2D, Input, Reshape, Permute, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
from keras.engine.topology import Layer
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input

import cv2
import numpy as np

class Softmax2D(Layer):
    def __init__(self, **kwargs):
        super(Softmax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return K.clip(e/s, 1e-7, 1)

    def get_output_shape_for(self, input_shape):
        return (input_shape)

class FullyConvolutionalNetwork():
    def __init__(self, batchsize=1, img_height=224, img_width=224, FCN_CLASSES=21):
        self.batchsize = batchsize
        self.img_height = img_height
        self.img_width = img_width
        self.FCN_CLASSES = FCN_CLASSES
        self.vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(3,self.img_height,self.img_width))

    def create_model(self, train_flag=True):
        #(samples, channels, rows, cols)
        ip = Input(shape=(3,self.img_height, self.img_width))
        h = self.vgg16.layers[1](ip)
        h = self.vgg16.layers[2](h)
        h = self.vgg16.layers[3](h)
        h = self.vgg16.layers[4](h)
        h = self.vgg16.layers[5](h)
        h = self.vgg16.layers[6](h)
        h = self.vgg16.layers[7](h)
        h = self.vgg16.layers[8](h)
        h = self.vgg16.layers[9](h)
        h = self.vgg16.layers[10](h)

        #split layer
        p3 = h
        p3 = Convolution2D(self.FCN_CLASSES, 1, 1, activation='relu', border_mode='valid')(p3)

        #(21*28*28)
        h = self.vgg16.layers[11](h)
        h = self.vgg16.layers[12](h)
        h = self.vgg16.layers[13](h)
        h = self.vgg16.layers[14](h)
        #(512*14*14)

        #split layer
        p4 = h
        p4 = Convolution2D(self.FCN_CLASSES, 1, 1, activation='relu')(p4)
        #"""
        p4 = Deconvolution2D(self.FCN_CLASSES, 4, 4,
                output_shape=(self.batchsize, self.FCN_CLASSES, 30, 30),
                subsample=(2, 2),
                border_mode='valid')(p4)
        p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)
        #"""
        #p4 = UpSampling2D((2,2))(p4)
        #p4 = Convolution2D(self.FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(p4)

        h = self.vgg16.layers[15](h)
        h = self.vgg16.layers[16](h)
        h = self.vgg16.layers[17](h)
        h = self.vgg16.layers[18](h)

        p5 = h
        p5 = Convolution2D(self.FCN_CLASSES, 1, 1, activation='relu')(p5)
        #"""
        p5 = Deconvolution2D(self.FCN_CLASSES, 8, 8,
                output_shape=(self.batchsize, self.FCN_CLASSES, 32, 32),
                subsample=(4, 4),
                border_mode='valid')(p5)
        p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
        #"""
        #p5 = UpSampling2D((4, 4))(p5)
        #p5 = Convolution2D(self.FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(p5)

        # merge scores
        h = merge([p3, p4, p5], mode="sum")

        #"""
        h = Deconvolution2D(self.FCN_CLASSES, 16, 16,
                output_shape=(self.batchsize, self.FCN_CLASSES, 232, 232),
                subsample=(8, 8),
                border_mode='valid')(h)
        #"""
        #h = UpSampling2D((8, 8))(h)
        
        #h = Convolution2D(self.FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(h)
        
        h = Cropping2D(cropping=((4, 4), (4, 4)))(h)
        
        #model_temp = Model(ip, h)
        #model_temp.summary()

        if not train_flag:
            return Model(ip, h)

        h = Reshape((self.FCN_CLASSES,self.img_height*self.img_width))(h)
        h = Permute((2,1))(h)
        out = Activation("softmax")(h)
        train_model = Model(ip, out)
        return train_model

if __name__ == "__main__":
    from keras.utils.visualize_util import model_to_dot, plot
    FCN = FullyConvolutionalNetwork()
    model, _ = FCN.create_model()
    plot(model, to_file='FCN_model.png',show_shapes=True)
