# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:20:58 2018

@author: Administrator
"""
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Input, Dense, concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalMaxPool2D
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
from sklearn.metrics import classification_report


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from collections import OrderedDict
import os
os.chdir("C:/Users/Administrator/Downloads")
from Cargar_imagenes_pano_bigimage_12sep import cargar_imagenes_bigpano
#############################################################################################
'''                         MODELOS A ENTRENAR KERAS :D                                   '''
#############################################################################################

def BGM_batch(vector_shape,n_categorias):
    Input_figure = Input(shape=vector_shape, name='input1')

    x = Conv2D(96, kernel_size=(3,3))(Input_figure)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(0.3)(x)
    
    
    x = Conv2D(256, kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(384, kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(500, kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = GlobalMaxPool2D()(x)
    
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    
    out = Dense(n_categorias, activation='sigmoid')(x)
    model = Model(inputs=Input_figure, outputs=out)
    return model

#############################################################################################
'''                               FUNCIONES DE REPORTE                                    '''
############################################################################################# 
# Cambiar zonas_singles por zonas_pano cuando corra todo con datos panoramicos

def graficos(history, x_test_img_z, y_test_z, zonaolugar, modelo, modelo_nombre):
    #history  = history_BGM3
    #modelo = 'BGM3'
    #zonaolugar = 'zonas_single'
    if zonaolugar == 'zonas_singles': 
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_singles/'
    elif zonaolugar == 'zonas_pano':
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_pano/'
    elif zonaolugar == 'lugares_singles':
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_singles/'
    else:
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_pano/'
             
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    archivo = directorio + modelo + '/' + modelo_nombre +'_acc.png'
    plt.savefig(archivo)
    plt.show()
    
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    archivo = directorio + modelo + '/' + modelo_nombre +'_loss.png'
    plt.savefig(archivo)
    plt.show()
    
import json
def guardar_history(history_var, zonaolugar, modelo, modelo_nombre):
    if zonaolugar == 'zonas_singles': 
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_singles/'
    elif zonaolugar == 'zonas_pano':
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_pano/'
    elif zonaolugar == 'lugares_singles':
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_singles/'
    else:
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_pano/'
        
    archivo = directorio + modelo + '/' + modelo_nombre +'_history'
    history_dict = history_var.history
    # Save it under the form of a json file
    json.dump(history_dict, open(archivo, 'w'))
    
def metricas(history, x_test_img_z, y_test_z, zonaolugar, modelo, modelo_nombre):   
    if zonaolugar == 'zonas_singles': 
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_singles/'
    elif zonaolugar == 'zonas_pano':
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_pano/'
    elif zonaolugar == 'lugares_singles':
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_singles/'
    else:
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_pano/'
    
    score = parallel_model.evaluate(x_test_img_z, y_test_z, verbose=1)
    pred = parallel_model.predict(x_test_img_z, batch_size=120, verbose=1)
    predicted = np.argmax(pred, axis=1)
    report = classification_report(np.argmax(y_test_z, axis=1), predicted)
    loss = 'Test loss:   ' + str(score[0])
    acc = 'Test accuracy:   ' + str(score[1])
    
    archivo = directorio + modelo + '/' + modelo_nombre +'_report.txt'
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
    
def guardar_weights(zonaolugar, modelo, modelo_nombre):   
    if zonaolugar == 'zonas_singles': 
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_singles/'
    elif zonaolugar == 'zonas_pano':
        directorio = 'C:/Users/Administrator/Documents/Resultados/zonas_pano/'
    elif zonaolugar == 'lugares_singles':
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_singles/'
    else:
        directorio = 'C:/Users/Administrator/Documents/Resultados/lugares_pano/'
    
    archivo = directorio + modelo + '/' + 'model_' + modelo_nombre +'_zonas_221x221_50ep.h5'
    parallel_model.save_weights(archivo)
    
def verificar_pred_correcta():
    pass
    
def guardar_top10_str(pred, y_test, idimagen, n_clases_mdl ,amazon, modelo):
    if amazon == True:
        directorio_top10 = 'C:/Users/Administrator/Documents/Resultados/top10'
    else:
        directorio_top10 = 'D:/Modelos Tensorflow/Conjunto train y test/top10'
        
    n_clases = n_clases_mdl 
    predicted = np.argmax(pred, axis=1)
    
    # En esta sección hay que agregar un True y un False para luego filtrar
    pred = np.insert(pred, (len(pred[0])), False, axis=1) #agregamos una columna con id nulo
    y_label_real = np.argmax(y_test, axis=1)
    for i in range(0,len(predicted)):
        valor = 'valores:'+ str(predicted[i]) + ' : ' + str(y_label_real[i])
        #print(valor)
        if predicted[i] == y_label_real[i]:
            #print('True')
            pred[i][(len(pred[0])) -1] = True
        
    pred = np.insert(pred, (len(pred[0])), 1, axis=1) #agregamos una columna con id nulo
    id_num = (len(pred[0])) -1
    for i in range(0,len(pred)):
        # Aca tenemos que agregar el id corespondiente
        pred[i][id_num] = idimagen[i]
    
    id_top = []
    proba_top = []
    nid = len(pred[0])-1
    for clase in range(0,n_clases):
        pred_orderbyclase = pred[pred[:,clase].argsort()[::-1]] #ordenamos de menor a mayor los resultados de la columna 1   
        array_top10 = []
        array_p10 = []
        top10 = 0
        count = 0
        valor = 'clase: ' + str(clase)
        print(valor)
        while top10 <= 10:
            print(pred_orderbyclase[count][len(pred[0])-2])
            if pred_orderbyclase[count][len(pred[0])-2] == True:
                print('inside')
                array_top10.append(int(pred_orderbyclase[count][nid]))
                array_p10.append(pred_orderbyclase[count][clase])
                top10 = top10 + 1
                count = count + 1
            else:
                count = count + 1
        id_top.append(array_top10)    
        proba_top.append(array_p10)  
        guardar_top = directorio_top10 + '/id_top_' +  modelo + '.txt'
        guardar_probatop = directorio_top10 + '/proba_top_' +  modelo + '.txt'
        
        np.savetxt(guardar_top, id_top, fmt="%d")
        np.savetxt(guardar_probatop, proba_top, fmt="%d")
    return id_top, proba_top

def guardar_top10_images(id_top,proba_top, amazon, modelo):
    if amazon == True:
        directorio_fotos = 'C:/Users/Administrator/Documents/Imagenes'
        directorio_top10 = 'C:/Users/Administrator/Documents/Resultados/top10'
    else:
        directorio_fotos = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/'
        directorio_top10 = 'D:/Modelos Tensorflow/Conjunto train y test/top10'
    try:
        for i in range(0,len(id_top)):
            for j in range(0,len(id_top[0])):
                id_img = id_top[i][j]
                #proba_img = round(id_top[i][j],3)
                if amazon == True:
                    f = str(int(id_img)) + '/gsv_pano.jpg'
                else:
                    f = str(int(id_img)) + '/0/gsv_0.jpg'
                directorio_archivo = os.path.join(directorio_fotos, f)
                img = Image.open(directorio_archivo)
                guardar_en = directorio_top10 + '/' + str(id_img) + '_' +  modelo + '_clase' + str(i) + '.jpg'
                img.save(guardar_en)  
    except:
        pass


#############################################################################################
'''                                      MODELO BGM3_sep                                  '''
#############################################################################################

total = len(y_train_z)
cantidad = sum(y_train_z)
nopesos = (cantidad/total)
    
sum(y_train_z)
           
                                                             
amazon = True
optimizador_amazon = 'rmsprop'
rgb = 3
x_train_img_z, idimagenes_train_z, y_train_z, x_test_img_z, idimagenes_test_z, y_test_z, n_categorias = cargar_imagenes_bigpano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar', prueba = False, amazon = True, balance= True, contraste = True, channel=True, augmentation = False)
if amazon == True:
    #optimizador_amazon = 'adam'
    optimizador_amazon = 'rmsprop'
    #####################################################################
    #                         BGM BATCH AMAZON                           #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM_batch = BGM_batch(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM_batch, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_BGM_batch = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 100,
                              epochs=30,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              )
    
    graficos(history_BGM_batch, x_test_img_z, y_test_z, 'lugares_singles', 'BGM4','BGM_batch_BIGpano_BR')
    guardar_history(history_BGM_batch, 'lugares_singles', 'BGM4','BGM_batch_BIGpano_BR')
    metricas(history_BGM_batch, x_test_img_z, y_test_z, 'lugares_singles', 'BGM4','BGM_batch_BIGpano_BR')
    guardar_weights('lugares_singles', 'BGM4','BGM_batch_BIGpano_BR')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM_batch_BIGpano_BR') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM_batch_BIGpano_BR') #el false es para  amazon
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM_batch_BIGpano_BR') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM_batch_BIGpano_BR') #el false es para  amazon
else:
    #####################################################################
    #                         BMG BATCH         PC                     #
    #####################################################################
    rgb = 3
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM_batch = BGM_batch(vector_shape, n_categorias)
    model_BGM_batch.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    
    history_BGM_batch = model_BGM_batch.fit(x_train_img_z, y_train_z,
                              batch_size = 20,
                              epochs=2,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              )
    
    archivo = 'D:/Modelos Tensorflow/Conjunto train y test/history_prueba/BGM3'
    
    pred = model_BGM_batch.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM3_ma2_ran') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM3_ma2_ran') #el false es para  amazon
    
    history_dict = history_BGM_batch.history
    # Save it under the form of a json file
    json.dump(history_dict, open(archivo, 'w'))
    