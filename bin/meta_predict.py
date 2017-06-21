from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.svm import SVR
import json
import numpy as np
import os
import h5py
import util

def run():
    f = open('/home/kirayue/final/metadata', 'r')
    metadata = json.load(f)
    label = open('dataset/t1_train_label.txt', 'r').readlines()

    X = []; y = []
    for ind, v in enumerate(label):
        if str(ind) in metadata:
            feature = []
            cur_metadata = metadata[str(ind)]
            feature.append(cur_metadata['commentCount'])
            feature.append(cur_metadata['num_groups'])
            feature.append(cur_metadata['viewCount'])
            feature.append(cur_metadata['faveCount'])
            X.append(feature)
            y.append(float(label[ind].replace('\n', '')))

    train_num = int(len(X) * 0.8)
    reg = RandomForestRegressor(n_estimators=400, n_jobs=50)
    print("Training ...")
    reg.fit(X[:train_num], y[:train_num])
    y_t = reg.predict(X[train_num:])
    print("MAE: {}".format(mean_absolute_error(y[train_num:], y_t)))
    print("MSE: {}".format(mean_squared_error(y[train_num:], y_t)))
    util.write_csv(y_t, y[train_num:], 'result/meta_predict.csv')

if __name__ == '__main__':
    run()

