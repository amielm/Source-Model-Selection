# A word of caution!!! the code may take a while to run depending on the number of cores on the computer. !!!
## need to run to avoid multiprocessing errors:
##  export OMP_NUM_THREADS=1

# Disclaimer: preprocessing code is based on the code from: https://github.com/hfawaz/bigdata18
import numpy as np
import pandas as pd
from os import listdir, walk
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shap
from matplotlib import pyplot as plt
from itertools import product
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count
import multiprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import ChainMap
from itertools import product

# function to read ucr dataset file from disk
def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

# function to build basic network architecture
def build_model(input_shape, nb_classes, pre_model_weights=None, sess=None):
    from keras.models import load_model
    import keras
    import keras.backend as K
    K.set_session(sess)

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    if pre_model_weights is not None:
        model.set_weights(pre_model_weights)

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(),
        metrics=['accuracy'])

    return model

def load_weights_to_premade_model(model, pre_model_weights=None):
    if pre_model_weights is not None:
        model.set_weights(pre_model_weights)
    return model

# read train and test from UCR dataset and save in dataset dictionary
def read_dataset(dataset_name, root_dir):
    dataset_dict = {}
    if dataset_name not in ["Fused_datasets"]:
        x_train, y_train = readucr(join(root_dir,dataset_name,"{}_TRAIN".format(dataset_name)))
        x_test, y_test = readucr(join(root_dir,dataset_name,"{}_TEST".format(dataset_name)))

        dataset_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())
    return dataset_dict

def read_all_datasets(root_dir):
    datasets_dict = {}
    # for dataset_name in listdir(root_dir):
    #     if dataset_name not in ["Fused_datasets"]:
    #         x_train, y_train = readucr(join(root_dir,dataset_name,"{}_TRAIN".format(dataset_name)))
    #         x_test, y_test = readucr(join(root_dir,dataset_name,"{}_TEST".format(dataset_name)))
    #
    #         datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
    #                 y_test.copy())

    results = []
    with Pool(8) as pool:
        results = pool.starmap(read_dataset, zip(listdir(root_dir), [root_dir]*len(listdir(root_dir))))
    datasets_dict = dict(ChainMap(*results))

    return datasets_dict

def transform_labels(y_train,y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train,y_test),axis =0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

def read_data_from_dataset(dataset_name, base_path):
    dataset_dict = read_dataset(dataset_name, base_path)

    x_train = dataset_dict[dataset_name][0]
    y_train = dataset_dict[dataset_name][1]
    x_test = dataset_dict[dataset_name][2]
    y_test = dataset_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    classes, classes_counts = np.unique(y_train, return_counts=True)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, nb_classes, classes

# get the activations of: "model(x)" at input to "layer"
def map2layer(model, x, layer, sess):

    feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
    return sess.run(model.layers[layer].input, feed_dict)

# datasets_dict = read_all_datasets(base_path)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

start_time= time.time()

# function to calculate embeddings for a given target dataset using a given source model
def do_transfer(target_tuple):
    start_time = time.time()
    from keras.models import load_model
    import keras
    import keras.backend as K
    import tensorflow as tf
    import os
    from sklearn.metrics import log_loss

    base_path = "UCR_TS_Archive_2015/"
    seed = 10
    cutoff_layer = -1

	# set-up for multiprocessing
    config = tf.ConfigProto(intra_op_parallelism_threads=128, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 128 })
    session = tf.Session(config=config)
    K.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "128"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    
    source_dataset_ind, source_dataset_name = target_tuple[1][0], target_tuple[1][1]
    target_dataset_ind, target_dataset_name = target_tuple[0][0], target_tuple[0][1]

	# extract training data for target dataset
    target_x_train, target_y_train, _, _, target_nb_classes, classes = read_data_from_dataset(target_dataset_name, base_path)
    # Only need the number of classes to build a model with the right shape output layer.
	# This is necessary only to simplify loading the source weights into the model.
	_, _, _, _, source_nb_classes, source_classes = read_data_from_dataset(source_dataset_name, base_path)

    model = build_model(target_x_train.shape[1:], source_nb_classes, None, K.get_session())
    model.load_weights(join(base_path, source_dataset_name, "best_model.hdf5"))

    target_x_sample = target_x_train
    target_y_sample = target_y_train
	# calculate embeddings for target_x_sample using the truncated model
    encodings = map2layer(model, target_x_sample, cutoff_layer, K.get_session())

    print("done with {} --> {} in {} seconds. #input {}, shap_shape:{}".format(source_dataset_ind,
                                                                                target_dataset_ind,
                                                                                time.time()-start_time,
                                                                                target_x_train.shape,
                                                                                encodings.shape))
    #return {(source_dataset_ind, target_dataset_ind): cur_sum_abs_shap}
    return (source_dataset_ind, target_dataset_ind, (encodings, target_y_sample))
    # shap_vectors[target_dataset_ind, target_dataset_ind] = cur_sum_abs_shap

if __name__ == "__main__":
	# use the fawaz's results to grab all the dataset names
    similarity_lists = pd.read_csv("bigdata18/results/similar_datasets.csv", index_col=0)
    sorted_unique_datasets = similarity_lists.index.unique().sort_values()
    num_datasets = len(sorted_unique_datasets)

	# code block to allow resuming code if interuppted in the middle... :-(
    load_path = "encoding_transfer.pkl"
    try:
        with open(load_path, "rb") as f:
            shap_vectors = pickle.load(f)

        for max_col in range(shap_vectors.shape[1]):
            try:
                _ = shap_vectors[-1, max_col].shape
            except:
                break
        print("resuming from {}".format(max_col))
        time.sleep(2)
    except:
        print("File doesn't exist")
        time.sleep(3)
        shap_vectors = np.empty((num_datasets, num_datasets), dtype=object)
        max_col =0

    save_path = load_path

    start_time = time.time()

	# run all of the encoding calculations in parallel
    target_source_tuple = list(product(list(enumerate(sorted_unique_datasets))[max_col:], list(enumerate(sorted_unique_datasets))))
    with Pool(28, maxtasksperchild=20) as pool:

        for s,t,results in pool.imap(do_transfer, target_source_tuple):
            shap_vectors[s,t] = results
            print("added {}-->{}".format(s,t))
            if s==(len(sorted_unique_datasets)-1):
                save_object(shap_vectors, save_path)
                print("finished and saved target {}".format(t))


    print("total time {} seconds".format(time.time()-start_time))

