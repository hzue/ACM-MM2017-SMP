from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from minepy import MINE
import numpy as np
import os
import h5py
import util

def run():
    f1 = h5py.File('image_features_4096.hdf5', 'r')
    f2 = h5py.File('image_arr_N_224_224_3.hdf5', 'r')
    X = f1['X'][()]
    y = f2['y'][:len(X)]

    model_path = 'model/rf_8000.joblib.pkl'
    reg = None
    if os.path.exists(model_path):
        reg = joblib.load(model_path)
    else:
        train_num = int(len(X) * 0.8)
        reg = RandomForestRegressor(n_estimators=400, n_jobs=50)
        print("Training ...")
        reg.fit(X[:train_num], y[:train_num])
        _ = joblib.dump(reg, model_path)

    y_t = reg.predict(X[300000:])
    print("MAE: {}".format(mean_absolute_error(y[300000:], y_t)))
    print("MSE: {}".format(mean_squared_error(y[300000:], y_t)))
    util.write_csv(y_t, y[300000:], 'result/rf_all.csv')

if __name__ == "__main__":
    run()
