import numpy as np
from PIL import Image
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image


print('K.image_dim_ordering: '+str(K.image_dim_ordering()))

def binarylab(labels, size, nb_class):
    y = np.zeros((size,size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i, j, labels[i][j]] = 1
    return y

def load_data(path, size, label=True):
    img = Image.open(path)
    w,h = img.size
    if w < h:
        if w < size:
            img = img.resize((size, size*h//w))
            w, h = img.size
    else:
        if h < size:
            img = img.resize((size*w//h, size))
            w, h = img.size
    
    #img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))
    img = img.resize((size, size))
    
    if label:
        y = np.array(img, dtype=np.int32)
        mask = y == 255
        y[mask] = 0
        y = binarylab(y, size, 21)
        y = np.expand_dims(y, axis=0)
        y = y.reshape((1,size*size,21))
        return y
    else:
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X)
        return X

def generate_arrays_from_file(names, path_to_train, path_to_target, img_size, nb_class):
    while True:
        for name in names:
            Xpath = path_to_train + "{}.jpg".format(name)
            ypath = path_to_target + "{}.png".format(name)
            X = load_data(Xpath, img_size, label=False)
            y = load_data(ypath, img_size, label=True)
            yield (X, y)
