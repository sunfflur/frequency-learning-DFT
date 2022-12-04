# -*- coding: utf-8 -*-

"""Experiment EyeQ database

In this experiment, two classes of the EyeQ dataset were explored: good and reject.

Considering these two classes, 22,358 images were used for training and evaluating
the models, according to the following division:
----- Training set – 8,347 images of good category and 2,320 images of reject category;
----- Validation set – 8,471 images of good category and 3,220 images of reject category.

~~Author: github.com/sunfflur~~

"""


#---#


""" Libraries Import """

s = 23
import os
os.environ['PYTHONHASHSEED']=str(s)
import random
random.seed(s)
from numpy.random import seed
seed(s)
import tensorflow as tf
#import tensorflow
tf.random.set_seed(s)
#from tensorflow.random import set_seed
#set_seed(s)

import PIL
import numpy as np

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.layers import Layer, Dense, Conv1D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json

import seaborn as sn
import tensorflow_addons as tfa
import tensorflow_recommenders as tfrs
import image_functions, data
from data import *
from custom_layers import FreqLayer
from plots import *
import time
import GPUtil
from psutil import virtual_memory
from tabulate import tabulate


#---#


"""#### GPU INFO """

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print("="*40, "GPU Details", "="*40)
gpus = GPUtil.getGPUs()
list_gpus = []
for gpu in gpus:
    # get the GPU id
    gpu_id = gpu.id
    # name of GPU
    gpu_name = gpu.name
    # get % percentage of GPU usage of that GPU
    gpu_load = f"{gpu.load*100}%"
    # get free memory in MB format
    gpu_free_memory = f"{gpu.memoryFree}MB"
    # get used memory
    gpu_used_memory = f"{gpu.memoryUsed}MB"
    # get total memory
    gpu_total_memory = f"{gpu.memoryTotal}MB"
    # get GPU temperature in Celsius
    gpu_temperature = f"{gpu.temperature} °C"
    gpu_uuid = gpu.uuid
    list_gpus.append((
        gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
        gpu_total_memory, gpu_temperature, gpu_uuid
    ))
print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature", "uuid")))

ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')


#---#



""" Image load """

### data loading ### 

path_train1 = 'G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\xm_p1.npy'
path_train2 = 'G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\xm_p2.npy'

path_test1 = 'G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\xmt_p1.npy'
path_test2 = 'G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\xmt_p2.npy'


#---#


"""#### Data Load"""

### Saved Pre-Processed data LOAD ###

input_train, y_train, input_test, y_test = Load(path_train='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\input_train_eyeq_l2_w1_square512_fft.npy',
                                                y_train='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\y_train.npy',
                                                path_valid=False,
                                                y_valid=False,
                                                path_test='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\input_test_eyeq_l2_w1_square512_fft.npy',
                                                #path_test='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\input_test_eyeq_l2_w1_square512_fht.npy',
                                                y_test='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\y_test.npy',
                                                #y_test='G:\\Meu Drive\\Mestrado\\Experimentos\\exp-eyeq\\exp-eyeq-data\\square-grouping\\3-levels\\y_test.npy',
                                                extra_layers='Dense')

""" Model"""

test_accuracies = []
total_times = []
mean = []
t_mean = []

optimizers = ['Adam']
for index, opt in enumerate(optimizers):
    print('>>> current optimizer:', opt)
    acc = []
    tempos = []
    for m in range(0, 1):
        print(">>>>>>>>>> test:", m)
        ### reset session ###
        tf.keras.backend.clear_session()
        ### dense ###
        tf.random.set_seed(s) #s
        init = 'glorot_normal' #glorot_normal
        function = 'LeakyReLU' #LeakyReLU
        model = Sequential([
        FreqLayer(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)), #0.01
        Dense(64, activation=function, #128 original
              kernel_initializer=init, bias_initializer=init),
        Dropout(0.25), #0.25
        Dense(32, activation=function, #128
             kernel_initializer=init, bias_initializer=init),
        Dropout(0.15),  #0.15
        Dense(32, activation=function, 
             kernel_initializer=init, bias_initializer=init),    
        Dense(2, activation='softmax')])
        
        """ Optimizers """

        ### inverse time decay  ###
        bs=32
        inversetime_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
          initial_learning_rate = 0.01, #0.01
          decay_steps = input_train.shape[0]/bs,
          decay_rate = 0.01) #0.01

        ### optimizers ###
        d_m=0.9 #momentum - dense layers 0.8

        if opt == 'Adam':
            opt = Adam(learning_rate=0.001)
        elif opt == 'SGD':
            opt = tfrs.experimental.optimizers.CompositeOptimizer([
                  (SGD(learning_rate=inversetime_decay, momentum=0.0), lambda: [model.layers[0].kernel]),
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[1].weights),
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[2].weights),
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[3].weights),
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[4].weights),
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[5].weights),   
                  (SGD(learning_rate=inversetime_decay, momentum=d_m), lambda: model.layers[6].weights)])
        
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        inicio = time.time()
        history =  model.fit(input_train, y_train, epochs=700, batch_size=bs, verbose=2, shuffle=True, validation_split=0.1)
        fim = time.time()
        tempo = fim-inicio
        print("time:", tempo)

        ### TEST ACC ###
        scores = model.evaluate(input_test, y_test)
        print('\ntest %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
        acc.append(scores[1])
        tempos.append(tempo)
    accuracies = tf.convert_to_tensor(acc)
    t = tf.convert_to_tensor(tempos)
    mean.append(tf.reduce_mean(accuracies))
    t_mean.append(tf.reduce_mean(t))
    test_accuracies.append(accuracies)
    total_times.append(t)
    
print('test_accuracies:', test_accuracies) #SGD and Adam
print('mean accuracy:', mean)
print('total_time:', total_times) #SGD and Adam
print('mean time:', t_mean)


#---#


""" TRAIN/VALIDATION EVOLUTION PLOT """

evolution_curves_plot(history=history, language='en')

""" CONFUSION MATRIX PLOT """

confusion_matrix_plot(data=input_test, y_test=y_test, model=model, language='en')

""" INCORRECT CLASS PLOT """

incorrect_class_plot(data=input_test, x_test=x_test, y_test=y_test, model=model)

""" FREQUENCY LAYER WEIGHTS PLOT """

freq_weights_plot(model=model)


print(model.summary())
