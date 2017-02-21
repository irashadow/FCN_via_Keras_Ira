from model import FullyConvolutionalNetwork
from preprocess import *
from keras.models import model_from_json
from PIL import Image
import numpy as np

def predict():
    FCN = FullyConvolutionalNetwork()
    model = FCN.create_model(train_flag=False)
    model.load_weights("./weights/fcn_params")
    X = load_data("demo_imgs/X1.jpg", 224, label=False)
    pred = model.predict(X)[0]
    pred = pred.argmax(axis=-1).astype(np.int32)
    img = Image.fromarray(pred)
    img.save("out.png")

predict()
