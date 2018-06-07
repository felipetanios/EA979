from __future__ import print_function
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy
from torchsample import TensorDataset
# import keras
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import csv
import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
from tqdm import tqdm
import dictionary
from sklearn.model_selection import train_test_split

import os
from torchvision import datasets


#################################
#    Loading Dictionary
#################################

dic_file = 'dic.p'
galaxy_dic = {}
galaxy_dic = dictionary.load_obj(dic_file)
#print(galaxy_dic)


#################################
#    Generating Histogram
#################################

histogram = dictionary.dataset_class_histogram(galaxy_dic)
#print(histogram)

#################################
#    Reducing Classes to

#    Sx : Spiral
   
#    S0 : Lenticular
   
#    Ix: Irregular
   
#    Ex : Elliptical
#################################


#defining morfology
for key,value in galaxy_dic.items():
    if value == "S0" or value == "S0-a":
        galaxy_dic.update({key: 'L'})

for key,value in galaxy_dic.items():
    if value[0] == "S" and value[1] == "B":
        galaxy_dic.update({key: 'SB'})
        
for key,value in galaxy_dic.items():
    if value[0] == "S" and value[1] != "B":
        galaxy_dic.update({key: 'S'})
        
for key,value in galaxy_dic.items():
    if value[0] == "I":
        galaxy_dic.update({key: 'I'})
        
for key,value in galaxy_dic.items():
    if value[0] == "E":
        galaxy_dic.update({key: 'E'}) 

for key,value in galaxy_dic.items():
    if value[0] == "|":
        galaxy_dic = dictionary.removekey(galaxy_dic,key)
    
histogram = dictionary.dataset_class_histogram(galaxy_dic)

print(histogram)

#################################
#     Creating Labels dictionary
#################################

i = 0
label_dic = {}
for k in galaxy_dic.values():
  if k not in label_dic.keys():
    label_dic.update({k : i})
    i = i + 1

number_of_labels = len(label_dic.keys())


#########
#################################
#    Creating Test and Train Sets
#################################
#########




#########
#Separating all set into single type subset
#########

L_galaxy_dic = []
for key,value in galaxy_dic.items():
    if value == "L":
        L_galaxy_dic.append(key)

L_x_train_names, L_x_test_names = train_test_split(L_galaxy_dic, test_size=0.4)

SB_galaxy_dic= []
for key,value in galaxy_dic.items():
    if value == "SB":
        SB_galaxy_dic.append(key)

SB_x_train_names, SB_x_test_names = train_test_split(SB_galaxy_dic, test_size=0.4)

S_galaxy_dic= []
for key,value in galaxy_dic.items():
    if value == "S":
        S_galaxy_dic.append(key)

S_x_train_names, S_x_test_names = train_test_split(S_galaxy_dic, test_size=0.4)


I_galaxy_dic= []
for key,value in galaxy_dic.items():
    if value == "I":
        I_galaxy_dic.append(key)

I_x_train_names, I_x_test_names = train_test_split(I_galaxy_dic, test_size=0.4)

E_galaxy_dic= []
for key,value in galaxy_dic.items():
    if value == "E":
        E_galaxy_dic.append(key)

E_x_train_names, E_x_test_names = train_test_split(E_galaxy_dic, test_size=0.4)

# x_train_names = subset with names of train files
# x_test_names = subset with names of test files

x_train_names = L_x_train_names + SB_x_train_names + S_x_train_names + I_x_train_names + E_x_train_names

x_test_names = L_x_test_names + SB_x_test_names + S_x_test_names + I_x_test_names + E_x_test_names



#########
#Gathering Train and Test Features
#########

i = 0
features_train = []
features_test = []
labels_train = []
labels_test = []

for path_to_image in tqdm(glob.glob("./images/png-grey/*.png")):
    # print(path_to_image[18:-4])
    name = path_to_image[18:-4]
    if name in x_train_names:
      im = imageio.imread(path_to_image)
      features_train.append(im)
      # label_train = [0]*number_of_labels
      idx = label_dic[galaxy_dic[name]]
      label_train = idx
      labels_train.append(label_train)


    if name in x_test_names:
      im = imageio.imread(path_to_image)
      features_test.append(im)
      # label_test = [0]*number_of_labels
      idx = label_dic[galaxy_dic[name]]
      label_test = idx
      labels_test.append(label_test)

features_train = np.array(features_train)
features_test = np.array(features_test)
# labels_train = np.array(labels_train)
# labels_test = np.array(labels_test)


x_train = features_train
x_test = features_test
y_train = labels_train
y_test = labels_test



train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32)
val_dataset = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define your model EXACTLY as if you were using nn.Module
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Network()
trainer = ModuleTrainer(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                L2Regularizer(scale=1e-5, module_filter='fc*')]
constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*')]
initializers = [XavierUniform(bias=False, module_filter='fc*')]
metrics = [CategoricalAccuracy(top_k=3)]

trainer.compile(loss='nll_loss',
                optimizer='adadelta',
                regularizers=regularizers,
                constraints=constraints,
                initializers=initializers,
                metrics=metrics, 
                callbacks=callbacks)

trainer.fit_loader(train_loader, val_loader, num_epoch=20, verbose=1)

