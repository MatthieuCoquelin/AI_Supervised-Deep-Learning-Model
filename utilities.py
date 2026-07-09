import h5py
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # train set features
    y_train = np.array(train_dataset["Y_train"][:]) # train set labels
    
    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # train set features
    y_test = np.array(test_dataset["Y_test"][:]) # train set labels

    return X_train, y_train, X_test, y_test

def load_tensorFlow_dataset(str_dataset):
    X_train, y_train = tfds.load(str_dataset, split=['train'], batch_size=-1, shuffle_files=False, as_supervised=True)[0]
    X_test, y_test = tfds.load(str_dataset, split=['test'], batch_size=-1, shuffle_files = False, as_supervised=True)[0]
    
    return X_train, y_train, X_test, y_test

# def load_tensorFlow_dataset_info(str_dataset):
#     (ds_train, ds_test), ds_info = tfds.load(str_dataset, split=["train", "test"], shuffle_files=True, as_supervised=False, with_info=True)

#     return X_train, y_train, X_test, y_test, info_train


##########################################################
#                    PREPROCESSING                       #
##########################################################

def Dataset_To_GrayScale(X_train, X_test): 
    X_train = tf.keras.ops.image.rgb_to_grayscale (X_train)   
    X_test = tf.keras.ops.image.rgb_to_grayscale (X_test)
    return X_train, X_test

def Resize_Dataset(X_train, X_test, Train_Scale, Test_Scale):
    X_train = tf.image.resize (X_train, [Train_Scale, Train_Scale])
    X_test = tf.image.resize (X_test, [Test_Scale, Test_Scale])
    return X_train, X_test

def Convert_TensorFlow_To_Numpy(X_train, y_train, X_test, y_test):
    X_train = tfds.as_numpy(X_train)
    X_test = tfds.as_numpy(X_test)
    y_train = tfds.as_numpy(y_train)
    y_test = tfds.as_numpy(y_test)

    return X_train, y_train, X_test, y_test