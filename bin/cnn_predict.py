from pprint import pprint
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import numpy as np
import h5py
import util

def run():
    data = h5py.File("image_arr_N_224_224_3.processed.hdf5", "r")
    test_X = data['X'][300000:]
    test_y = data['y'][300000:]
    model = load_model('model/fine_tune_VGG19.train_conv.train_fc.model')
    y_t = model.predict(test_X)
    print("MAE: {}".format(mean_absolute_error(test_y, y_t)))
    print("MSE: {}".format(mean_squared_error(test_y, y_t)))
    util.write_csv(y_t, test_y, 'result/cnn.csv')

if __name__ == "__main__":
    run()
