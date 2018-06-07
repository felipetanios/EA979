import csv
import pickle
import os
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import copy
from scipy.stats import friedmanchisquare as friedman
from scipy.stats import wilcoxon as wilcoxon
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_ind_from_stats as ttest2
from scipy.stats import f_oneway as f_oneway
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


dic_file = 'dic.p'
galaxy_dic = {}

galaxy_dic = load_obj(dic_file)
print(galaxy_dic)

histogram = dataset_class_histogram(galaxy_dic)
print(histogram)

galaxy_dic_aux = galaxy_dic


#defining lenticular
for key,value in galaxy_dic_aux.items():
    if value == "S0" or value == "S0-a":
        galaxy_dic_aux.update({key: 'L'})

for key,value in galaxy_dic_aux.items():
    if value[0] == "S" and value[1] == "B":
        galaxy_dic_aux.update({key: 'SB'})
        
for key,value in galaxy_dic_aux.items():
    if value[0] == "S" and value[1] != "B":
        galaxy_dic_aux.update({key: 'S'})
        
for key,value in galaxy_dic_aux.items():
    if value[0] == "I":
        galaxy_dic_aux.update({key: 'I'})
        
for key,value in galaxy_dic_aux.items():
    if value[0] == "E":
        galaxy_dic_aux.update({key: 'E'}) 

for key,value in galaxy_dic_aux.items():
    if value[0] == "|":
        galaxy_dic_aux = removekey(galaxy_dic_aux,key)
    
histogram = dataset_class_histogram(galaxy_dic_aux)
print(histogram)

path = './images/png-grey/'

all_keys = galaxy_dic_aux.keys()

# print (all_keys)

for path_to_image in tqdm(glob.glob("./images/png/*.png")):
#     print (path_to_image[13:23])
    if path_to_image[13:23] in all_keys:

        im = imageio.imread(path_to_image)
        new_path = path+path_to_image[13:]

        greyscale = im
        for i in range(len(im)):
            for j in range(len(im[i])):
                for k in range(len(im[i][j])):
                    greyscale[i][j][k] = im[i][j][0]

        imageio.imwrite(new_path, greyscale)
