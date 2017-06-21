from keras.preprocessing import image
from pprint import pprint
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os
import h5py

def run():
    X = []; y = []; count = 0
    all_labels = open('dataset/t1_train_label.txt').readlines()

    for img_path in os.listdir('image/'):
        img = image.load_img("image/{}".format(img_path), target_size=(224,224))
        x = image.img_to_array(img)
        X.append(x)
        y.append(float(all_labels[int(img_path.replace(".jpg", ""))].replace("\n", "")))
        count += 1
        print("iter #: {}\r".format(count), end='')

    f1 = h5py.File("image_arr_N_224_224_3.hdf5", "w")
    f2 = h5py.File("image_arr_N_224_224_3.processed.hdf5", "w")

    X = np.asarray(X)

    f1.create_dataset("X", data=X)
    X = preprocess_input(X)
    f2.create_dataset("X", data=X)

    f1.create_dataset("y", data=y)
    f2.create_dataset("y", data=y)

    f1.close()
    f2.close()

if __name__ == "__main__":
    run()
