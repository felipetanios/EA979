from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

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
#Separating in train, validation and test
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


L_x_validation_names, L_x_test_names = train_test_split(L_x_test_names, test_size=0.5)
SB_x_validation_names, SB_x_test_names = train_test_split(SB_x_test_names, test_size=0.5)
S_x_validation_names, S_x_test_names = train_test_split(S_x_test_names, test_size=0.5)
I_x_validation_names, I_x_test_names = train_test_split(I_x_test_names, test_size=0.5)
E_x_validation_names, E_x_test_names = train_test_split(E_x_test_names, test_size=0.5)


x_train_names = L_x_train_names + SB_x_train_names + S_x_train_names + I_x_train_names + E_x_train_names
x_test_names = L_x_test_names + SB_x_test_names + S_x_test_names + I_x_test_names + E_x_test_names
x_validation_names = L_x_validation_names + SB_x_validation_names + S_x_validation_names + I_x_validation_names + E_x_validation_names


#########
#Gathering Train and Test Features
#########

i = 0
features_train = []
features_test = []
features_validation = []
labels_train = []
labels_test = []
labels_validation = []

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

    if name in x_validation_names:
      im = imageio.imread(path_to_image)
      features_validation.append(im)
      # label_test = [0]*number_of_labels
      idx = label_dic[galaxy_dic[name]]
      label_validation = idx
      labels_validation.append(label_validation)

features_train = np.array(features_train)
features_test = np.array(features_test)
features_validation = np.array(features_validation)
# labels_train = np.array(labels_train)
# labels_test = np.array(labels_test)


x_train = features_train
x_validation = features_validation
x_test = features_test
y_train = labels_train
y_validation = labels_validation
y_test = labels_test


#O numero de passos vai ser 50000/(batch_size)
batch_size = 64 #numero de amostras por gradiente
num_classes = number_of_labels # numero de classes do cifar10
epochs = 20 #numero de epocas para treinar o modelo
data_augmentation = False

# # The data, shuffled and split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()


## CHANGE TO APPEND FROM DICTIONARY

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)



print('y_train shape:', y_train.shape)
# print ('size of first dimention ', y_train[0][0])
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

model = Sequential()

#primeira mascara de convolucao
model.add(Conv2D(64, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# segunda mascara de convolucao
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



# initiate RMSprop optimizer
# RMS usa back propagation [RMS(w) eh em funcao de w-1] quadratico

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# ao invez de pegar todos os treinos e testes, o programa pega apenas uma porcao
x_train /= 255
x_test /= 255

## falta arrumar validation
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_validation, y_validation),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=x_train.shape[0] // batch_size, 
                        epochs=epochs, validation_data=(x_validation, y_validation))

score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
print("Resultado do teste final de acerto da rede")
print(score)

#keras get_layer: retorna a layer: podemos usar para ver o estado final dos filtros usados (ja que sao 5x5)
#visualize single neuron ->> output para apresentação
