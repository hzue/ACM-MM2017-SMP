from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import time
import numpy as np
import h5py
import os

def run():
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    print("1. read data")
    data = h5py.File("image_arr_N_224_224_3.processed.hdf5", "r")
    print("2. generate features")
    start_time = time.time()
    features = model.predict(np.asarray(data['X'][:]))
    print("--- %s seconds ---" % (time.time() - start_time))
    f = h5py.File("image_features_4096.hdf5", "w")
    f.create_dataset("X", data=features)
    f.close()
    data.close()

if __name__ == "__main__":
    run()
