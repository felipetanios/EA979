import csv
import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
from tqdm import tqdm


def save_obj(file_name, obj_name ):
    pickle.dump( obj_name, open( file_name, "wb" ) )

def load_obj(file_name):
    return pickle.load( open( file_name, "rb" ) )


def dataset_class_histogram(dataset):
    histogram = {}
    for data in dataset:
        if dataset[data] not in histogram:
            histogram[dataset[data]] = 1
        else:
            histogram[dataset[data]] += 1
    return histogram

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def labeling(dataset):
    lables = []
    for key in dataset:
        lables.append(dataset[key])
    return lables

def label_to_numbers(labels, d):
    dict_list = list(d)
    return [dict_list.index(i) for i in labels]

def numbers_to_labels(numbers, d):
    return [d.keys[i] for i in numbers]

def f1_from_confusion(c):
    ret = []
    for i in range(c.shape[0]):
        n_i = np.sum(c[i,:])
        est_i = np.sum(c[:,i])
        if n_i > 0:
            R = c[i,i] / float(n_i)
        else:
            R = 0.0
        if est_i > 0:
            P = c[i,i] / float(est_i)
        else:
            P = 0.0

        if (R+P) > 0:
            F = 2*R*P/(R+P)
        else:
            F = 0.
        ret.append([R, P, F])
    return ret