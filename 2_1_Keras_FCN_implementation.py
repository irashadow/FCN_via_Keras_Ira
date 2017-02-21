
# coding: utf-8

# In[76]:

import theano
from theano.sandbox import cuda
theano.sandbox.cuda.use("gpu0")


# In[243]:

#get_ipython().magic(u'matplotlib inline')

import model; reload(model)
from model import FullyConvolutionalNetwork

import preprocess; reload(preprocess)
from preprocess import *

import h5py

from keras.optimizers import Adam
from keras import backend as K


# In[244]:

print K.image_dim_ordering()


# In[245]:

img_size = 224
nb_class = 21
path_to_train = '/home/irashadow/python_workspace/data/PASCAL_VOC_2012/VOCdevkit/VOC2012/train/'
path_to_target = '/home/irashadow/python_workspace/data/PASCAL_VOC_2012/VOCdevkit/VOC2012/target/'
path_to_txt = '/home/irashadow/python_workspace/data/PASCAL_VOC_2012/VOCdevkit/VOC2012/names.txt'



# In[246]:

with open(path_to_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
nb_data = len(names)
print nb_data


# # FCN Model

# In[247]:

FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size, FCN_CLASSES=nb_class)


# In[248]:

train_model = FCN.create_model(train_flag=True)


# # Training Processing

# In[249]:

#Keras parameters
train_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#train_model.compile(loss="categorical_crossentropy",optimizer='adadelta',metrics=["accuracy"])
print("Num data: {}".format(nb_data))


# model preview

# In[250]:

train_model.summary()


# In[257]:

#Keras parameters
epoch = 100
batchsize = 1
train_model.optimizer.lr=1e-2


# In[258]:

from skimage import io
X = generate_arrays_from_file(names,path_to_train,path_to_target,img_size,nb_class)

X_get = X.next()
print X_get[0].shape
print X_get[1].shape



# In[259]:

train_model.fit_generator(generate_arrays_from_file(names,path_to_train,path_to_target,img_size, nb_class),
                          samples_per_epoch=nb_data,
                          nb_epoch=epoch)


# In[260]:


train_model.save_weights("weights/temp_voc_large", overwrite=True)
f = h5py.File("weights/temp_voc_large")


# In[261]:

layer_names = [name for name in f.attrs['layer_names']]
fcn = FCN.create_model(train_flag=False)

for i, layer in enumerate(fcn.layers):
    g = f[layer_names[i]]
    weights = [g[name] for name in g.attrs['weight_names']]
    layer.set_weights(weights)

fcn.save_weights("weights/fcn_params_pascal_voc_2012", overwrite=True)

f.close()
