from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import h5py

def run():
    data = h5py.File("image_arr_N_224_224_3.processed.hdf5", "r")
    base_model = VGG19(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers: layer.trainable = False
    print("base_model layer num: {}".format(len(base_model.layers)))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    for i in range(1, 31):
        X = np.asarray(data['X'][(i-1)*10000:i*10000])
        y = np.asarray(data['y'][(i-1)*10000:i*10000])
        print("---- start train batch {} ----".format(i))
        model.fit(X, y, epochs=1, batch_size=500)
        model.save('model/fine_tune_VGG19.model')

if __name__ == '__main__':
    run()
