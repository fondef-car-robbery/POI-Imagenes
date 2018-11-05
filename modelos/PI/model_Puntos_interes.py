# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:41:27 2018

@author: Bgm9
"""

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import scipy.misc
from skimage import data, io, filters
from PIL import Image
from keras.models import load_model
from keras.models import save_model
import h5py
from keras import optimizers
from random import shuffle
from resizeimage import resizeimage
from numpy import array

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from haversine import haversine
from keras.layers import Input, Dense, concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalMaxPool2D


#############################################################################################
'''                         FUNCIONES DE CARGA DE DATOS                                   '''
#############################################################################################

def data_PI(nombre_etiqueta, metodo_traintest):
    # nombre_etiqueta = 'label_lugar'
    # metodo_traintest = 'random'
    len_train = 10000
    df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
    df_PI = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header=0, encoding = 'latin1')
    
    if metodo_traintest == 'random':
        zona_riesgo = df_ids.loc[df_ids[nombre_etiqueta] == 1]
        zona_riesgo_subset = zona_riesgo.sample(n= int(len_train/2*1.2))
        train_1 = zona_riesgo_subset.sample(n=int(len_train/2))
        test_1 = zona_riesgo_subset[~zona_riesgo_subset.isin(train_1)].dropna()
        
        zona_noriesgo = df_ids.loc[df_ids[nombre_etiqueta] == 0]
        zona_noriesgo_subset = zona_noriesgo.sample(n= int(len_train/2*1.2))
        train_0 = zona_noriesgo_subset.sample(n=int(len_train/2))
        test_0 = zona_noriesgo_subset[~zona_noriesgo_subset.isin(train_0)].dropna()
        
        train = [train_1, train_0]
        test = [test_1, test_0]
        train = pd.concat(train)
        test = pd.concat(test)
        
    elif metodo_traintest == 'separar':
        x_plaza = -33.437914
        y_plaza = -70.650339
        n = random.randint(0, len(df_ids))
        
        #x_ale = df_ids.iloc[n]['x']
        #y_ale = df_ids.iloc[n]['y']
        x_ale = -33.383787 
        y_ale = -70.526801
        print(x_ale)
        print(y_ale)
        
        df_ids['train'] = 0
        m = (y_ale-y_plaza)/(x_ale-x_plaza)
        for i in range(0,len(df_ids)):
            y_recta = m*(df_ids.iloc[i]['x']-x_plaza)+y_plaza
            y_real = df_ids.iloc[i]['y']
            if y_real > y_recta:
                df_ids.set_value(i, 'train', 1)
    
    df_train = pd.merge(df_PI, train, on ='id', how='inner')
    df_test = pd.merge(df_PI, test,  on ='id', how='inner')
            
    df_train_PI = df_train.iloc[:,0:(len(df_train.columns)-7)]
    df_test_PI = df_test.iloc[:,0:(len(df_test.columns)-7)]
    
    df_train_PI['dist_plazarmas'] = ''
    df_train_PI['dist_costanera'] = ''
    df_test_PI['dist_plazarmas'] = ''
    df_test_PI['dist_costanera'] = ''
    xcostanera = -33.417313
    ycostanera = -70.606261
    xplaza = -33.437637
    yplaza = -70.650471
    for i in range(0,len(df_train_PI)):
        xi = df_train_PI.iloc[i]['x_x']
        yi = df_train_PI.iloc[i]['y_x']
        dist_costanera = haversine((xi,yi), (xcostanera,ycostanera))      
        dist_plazarmas = haversine((xi,yi), (xplaza,yplaza))     
        df_train_PI.at[i, 'dist_costanera'] = dist_costanera
        df_train_PI.at[i, 'dist_plazarmas'] = dist_plazarmas
    for i in range(0,len(df_test_PI)):
        xi = df_test_PI.iloc[i]['x_x']
        yi = df_test_PI.iloc[i]['y_x']
        dist_costanera = haversine((xi,yi), (xcostanera,ycostanera))      
        dist_plazarmas = haversine((xi,yi), (xplaza,yplaza))     
        df_test_PI.at[i, 'dist_costanera'] = dist_costanera
        df_test_PI.at[i, 'dist_plazarmas'] = dist_plazarmas
        
    df_train_PI = df_train_PI.iloc[:,3:len(df_train.columns)]
    df_test_PI = df_test_PI.iloc[:,3:len(df_test.columns)]
    
    
    df_train_label = df_train.loc[:,[nombre_etiqueta]]
    df_etiquetas =  df_train_label.values.tolist()
    num_classes = 2
    df_etiquetas_train = keras.utils.to_categorical(df_etiquetas, num_classes)
    
    df_test_label = df_test.loc[:,[nombre_etiqueta]]
    df_etiquetas =  df_test_label.values.tolist()
    num_classes = 2
    df_etiquetas_test = keras.utils.to_categorical(df_etiquetas, num_classes)
    
    #df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_final_train_test_XY_etiqueta_zonas.csv', header=0, encoding = 'latin1')
    #df_train_PI.column_names()
    
    del df_train_PI['x_y']
    del df_train_PI['y_y']
    del df_train_PI['fecha']

    del df_test_PI['x_y']
    del df_test_PI['y_y']
    del df_test_PI['fecha']
    ''' NORMALIZAR POR EL MAXIMO POR COLUMNA'''
    #for i in range(0,len(x_train.columns)):
    #    x_train.iloc[:,i] = x_train.iloc[:,i]/max(x_train.iloc[:,i])
    #    
    #for i in range(0,len(x_test.columns)):
    #    x_test.iloc[:,i] = x_test.iloc[:,i]/max(x_test.iloc[:,i])
    ''' NORMALIZAR POR EL MAXIMO DE LA BASE'''
    max_PI = 0
    for i in range(1,(len(df_train_PI.columns)-2)):
        if max(df_train_PI.iloc[:,i]) > max_PI:
            max_PI = max(df_train_PI.iloc[:,i])
            
    for i in range(1,(len(df_test_PI.columns)-2)):
        if max(df_test_PI.iloc[:,i]) > max_PI:
            max_PI = max(df_test_PI.iloc[:,i])
            
    for i in range(1,(len(df_test_PI.columns)-2)):
        df_train_PI.iloc[:,i] = df_train_PI.iloc[:,i]/max_PI
        df_test_PI.iloc[:,i] = df_test_PI.iloc[:,i]/max_PI
    
    #### 
    ### NUEVO
    ####
    
    #df_ids_etiqueta = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_final_train_test_XY_etiqueta_lugars.csv', header=0, encoding = 'latin1')

    
    '''
    df_ids.dropna(subset=[tipolabel], inplace=True)
    df_ids = df_ids.loc[:,[tipolabel]]
    df_ids = df_ids[tipolabel].astype('int').astype('str')
    df_ids =  df_ids.tolist()
    num_classes = 2
    df_etiqueta = keras.utils.to_categorical(df_ids, num_classes)
    '''
    
    return df_etiquetas_train, df_train_PI, df_etiquetas_test, df_test_PI 

y_train, x_train, y_test, x_test = data_PI(nombre_etiqueta = 'label_lugar', metodo_traintest = 'random')

def model_1():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim = 210))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))
    return model

def model_2():
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim = 210))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

def model_3():
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim = 210))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

def model_4():
    Input_angle = Input(shape=(210,), name = 'input2')

    y = Dense(500)(Input_angle)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Dropout(0.3)(y)
    y = Dense(500)(Input_angle)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Dropout(0.3)(y)
    out = Dense(2, activation='sigmoid')(y)
    model = Model(inputs=Input_angle, outputs=out)
    return model

#model = model_1()
#model = model_2()
#model = model_3()
model = model_4()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
              batch_size=50000,
              epochs=1000,
              verbose=1,
              shuffle = True,
              validation_data=(x_test, y_test))


directorio = 'C:/BGM - Contenido Memoria/Resultado modelos/PI'
modelo = 'BG_batch_PI'
modelo_nombre = 'BG_batch_PI_r30' 
####################################################################
#       GUARDAR GRAFICOS DEL MODELO
####################################################################
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
archivo = directorio + '/' + modelo_nombre +'_acc.png'
plt.savefig(archivo)
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
archivo = directorio + '/' + modelo_nombre +'_loss.png'
plt.savefig(archivo)
plt.show()

####################################################################
#       GUARDAR HISTORY DEL MODELO
####################################################################
import json
archivo = directorio + '/' + modelo_nombre +'_history'
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open(archivo, 'w'))


####################################################################
#       GUARDAR REPORT DEL MODELO
####################################################################
from sklearn.metrics import classification_report

score = model.evaluate(x_test, y_test, verbose=1)
pred = model.predict(x_test, batch_size=120, verbose=1)
predicted = np.argmax(pred, axis=1)
report = classification_report(np.argmax(y_test, axis=1), predicted)
loss = 'Test loss:   ' + str(score[0])
acc = 'Test accuracy:   ' + str(score[1])

archivo = directorio + '/' + modelo_nombre +'_report.txt'
text_file = open(archivo, "w")
text_file.write(report)
text_file.close()
text_file = open(archivo, "a")
text_file.write(loss)
text_file.close()
text_file = open(archivo, "a")
text_file.write(acc)
text_file.close()

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(modelo_nombre)

















