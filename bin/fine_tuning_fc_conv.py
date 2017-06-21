from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
import h5py

def run():
    data = h5py.File("image_arr_N_224_224_3.processed.hdf5", "r")
    model = load_model('model/fine_tune_VGG19.train_conv.model')
    for layer in model.layers[:22]: layer.trainable = False
    for layer in model.layers[22:]: layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error')

    for i in range(1, 31):
        X = np.asarray(data['X'][(i-1)*10000:i*10000])
        y = np.asarray(data['y'][(i-1)*10000:i*10000])
        print("---- start train batch {} ----".format(i))
        model.fit(X, y, epochs=1, batch_size=500)
        model.save('model/fine_tune_VGG19.train_conv.train_fc.model')

if __name__ == '__main__':
    run()
