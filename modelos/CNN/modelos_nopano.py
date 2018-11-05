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
from Cargar_imagenes_v6_11sep import cargar_imagenes_nopano, data_augmentation

#############################################################################################
'''                         MODELOS A ENTRENAR KERAS :D                                   '''
#############################################################################################
def BGM_3(vector_shape, n_categorias):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=vector_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (2, 2), activation='relu')) 
    model.add(Conv2D(50, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categorias, activation='sigmoid'))
    return model

def BGM_4(vector_shape, n_categorias):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=vector_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='relu')) 
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categorias, activation='sigmoid'))
    return model

def BGM_6(vector_shape, n_categorias):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=vector_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='relu')) 
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categorias, activation='sigmoid'))
    return model

def BGM_5(vector_shape, n_categorias):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=vector_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='relu')) 
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categorias, activation='sigmoid'))
    return model

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

def DENSENET121(vector_shape, n_categorias):
    #weights='imagenet'
    base_model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, pooling = 'avg', input_shape=vector_shape)
    last = base_model.output
    preds = Dense(n_categorias, activation='sigmoid')(last)
    model = Model(base_model.input, preds)
    model.load_weights('C:/Users/Administrator/Documents/Resultados/lugares_singles/BGM3/model_DENSENET121_ran_nopano_0a270_10ep_zonas_221x221_50ep.h5')
    return model

def RESNET50(vector_shape, n_categorias):
    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling = 'avg', input_shape=vector_shape)
    last = base_model.output
    preds = Dense(n_categorias, activation='sigmoid')(last)
    model = Model(base_model.input, preds)
    return model

def INCEPTIONV3(vector_shape, n_categorias):
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, pooling = 'avg', input_shape=vector_shape)
    last = base_model.output
    preds = Dense(n_categorias, activation='sigmoid')(last)
    model = Model(base_model.input, preds)
    return model

def EXCEPTION(vector_shape, n_categorias):
    base_model = keras.applications.xception.Xception(include_top=False, weights=None, pooling = 'avg', input_shape=vector_shape)
    last = base_model.output
    preds = Dense(n_categorias, activation='sigmoid')(last)
    model = Model(base_model.input, preds)
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
# Para probar mejora accuracy filtrando por mas valores
#x_train_img_z, idimagenes_train_z, y_train_z, x_test_img_z, idimagenes_test_z, y_test_z, n_categorias = cargar_imagenes_nopano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar', prueba = False, amazon = True, balance=True, contraste = True, channel=True, augmentation = False)

# Para probar mejora accuracy filtrando por mas valores y aumentando la data de entrenamiento (ojo con desbalance)
x_train_img_z, idimagenes_train_z, y_train_z, x_test_img_z, idimagenes_test_z, y_test_z, n_categorias = cargar_imagenes_nopano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar>2', prueba = False, amazon = True, balance= True, contraste = True, channel=True, augmentation = False)

x_train_augmentation, y_train_augmentation, id_train_augmentation = data_augmentation(n_categorias, idimagenes_train_z, nombre_etiqueta = 'label_lugar>2', amazon = False)
x_train_img_z = np.concatenate((x_train_img_z, x_train_augmentation))
idimagenes_train_z = np.concatenate((idimagenes_train_z, id_train_augmentation))
y_train_z = np.concatenate((y_train_z, y_train_augmentation))



# Para probar mejora accuracy filtrando por mas valores y aumentando la data de entrenamiento y las clases (ojo con desbalance)
#x_train_img_z, idimagenes_train_z, y_train_z, x_test_img_z, idimagenes_test_z, y_test_z, n_categorias = cargar_imagenes_nopano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar>2', prueba = False, amazon = True, balance=False, contraste = True, channel=True, augmentation = True)

total = len(y_train_z)
cantidad = sum(y_train_z)
nopesos = (cantidad/total)
    
sum(y_test_z)
class_weight = {0: 1,
                1: 1} 

class_weight = {0: 339,
                1: 48,
                2: 12,
                3: 1}  
    
            
                                                             
amazon = True
optimizador_amazon = 'rmsprop'
rgb = 3
if amazon == True:
    #optimizador_amazon = 'adam'
    optimizador_amazon = 'rmsprop'
    #####################################################################
    #                         BGM BATCH                             #
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
                              batch_size = 60,
                              epochs=30,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    
    graficos(history_BGM_batch, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM_batch_RNPLLSBNAU')
    guardar_history(history_BGM_batch, 'lugares_singles', 'BGM3','BGM_batch_RNPLLSBNAU')
    metricas(history_BGM_batch, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM_batch_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','BGM_batch_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM_batch_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM_batch_RNPLLSBNAU') #el false es para  amazon
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM_batch_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM_batch_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                         BGM3 MAXPOOL                              #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM3 = BGM_3(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM3, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_BGM3 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 110,
                              epochs=4,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    norobo = 0
    for i in range(0,len(y_train_z)):
        if y_train_z[i][0]==1:
            norobo = norobo + 1
    print(norobo)
    
    norobo = 0
    for i in range(0,len(pred)):
        if pred[i][0]>0.5:
            norobo = norobo + 1
    print(norobo)
    len(pred) - norobo
    
    graficos(history_BGM3, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM3_RNPLLSBNAU_4ep')
    guardar_history(history_BGM3, 'lugares_singles', 'BGM3','BGM3_RNPLLSBNAU_4ep')
    metricas(history_BGM3, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM3_RNPLLSBNAU_4ep')
    guardar_weights('lugares_singles', 'BGM3','BGM3_RNPLLSBNAU_4ep')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=100, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM3_RNPLLSBNAU_4ep') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM3_RNPLLSBNAU_4ep') #el false es para  amazon
    #####################################################################
    #                         BGM4 AVGPOOL                              #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM4 = BGM_4(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM4, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_BGM4 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 100,
                              epochs=20,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    graficos(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM4_lzona_RNPLLSBNAU')
    guardar_history(history_BGM4, 'lugares_singles', 'BGM3','BGM4_lzona_RNPLLSBNAU')
    metricas(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM4_lzona_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','BGM4_lzona_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM4_lzona_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM4_lzona_RNPLLSBNAU') #el false es para  amazon
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM4_lzona_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM4_lzona_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                         BGM5 MAXPOOL  4069                        #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM5 = BGM_5(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM5, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_BGM5 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 100,
                              epochs=5,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    graficos(history_BGM5, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM5_ran_nopano')
    guardar_history(history_BGM5, 'lugares_singles', 'BGM3','BGM5_ran_nopano')
    metricas(history_BGM5, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM5_ran_nopano')
    guardar_weights('lugares_singles', 'BGM3','BGM5_ran_nopano')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM5_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM5_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                         BGM6 AVGPOOL   4069                       #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM6 = BGM_6(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM6, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_BGM6 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 100,
                              epochs=4,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    graficos(history_BGM6, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM6_RNPLLSBNAU_4ep')
    guardar_history(history_BGM6, 'lugares_singles', 'BGM3','BGM6_RNPLLSBNAU_4ep')
    metricas(history_BGM6, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM6_RNPLLSBNAU_4ep')
    guardar_weights('lugares_singles', 'BGM3','BGM6_RNPLLSBNAU_4ep')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'BGM6_RNPLLSBNAU_4ep') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'BGM6_RNPLLSBNAU_4e') #el false es para  amazon
    #####################################################################
    #                            DENSENET121                            #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_DENSENET121 = DENSENET121(vector_shape, n_categorias)
      
    parallel_model = keras.utils.multi_gpu_model(model_DENSENET121, weight='Imagenet', gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])
    
    history_DENSENET121 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 45,
                              epochs=5,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
        
    graficos(history_DENSENET121, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','DENSENET121_RNPLLSBNAU')
    guardar_history(history_DENSENET121, 'lugares_singles', 'BGM3','DENSENET121_RNPLLSBNAU')
    metricas(history_DENSENET121, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','DENSENET121_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','DENSENET121_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'DENSENET121_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'DENSENET121_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                            RESNET50                               #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_RESNET50 = RESNET50(vector_shape, n_categorias)
      
    parallel_model = keras.utils.multi_gpu_model(model_RESNET50, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])

    history_RESNET50 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 45,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    
    graficos(history_RESNET50, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','RESNET50_RNPLLSBNAU')
    guardar_history(history_RESNET50, 'lugares_singles', 'BGM3','RESNET50_RNPLLSBNAU')
    metricas(history_RESNET50, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','RESNET50_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','RESNET50_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'RESNET50_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'RESNET50_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                           INCEPTIONV3                             #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_INCEPTIONV3 = INCEPTIONV3(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_INCEPTIONV3, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])

    history_INCEPTIONV3 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 5,
                              epochs=15,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    
    graficos(history_INCEPTIONV3, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','INCEPTIONV3_RNPLLSBNAU')
    guardar_history(history_INCEPTIONV3, 'lugares_singles', 'BGM3','INCEPTIONV3_RNPLLSBNAU')
    metricas(history_INCEPTIONV3, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','INCEPTIONV3_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','INCEPTIONV3_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'INCEPTIONV3_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'INCEPTIONV3_RNPLLSBNAU') #el false es para  amazon
    #####################################################################
    #                            EXCEPTION                              #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_EXCEPTION = EXCEPTION(vector_shape, n_categorias)
    
    parallel_model = keras.utils.multi_gpu_model(model_EXCEPTION, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=optimizador_amazon,
                           metrics=['accuracy'])

    history_EXCEPTION = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 30,
                              epochs=15,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    
    graficos(history_EXCEPTION, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','EXCEPTION_RNPLLSBNAU')
    guardar_history(history_EXCEPTION, 'lugares_singles', 'BGM3','EXCEPTION_RNPLLSBNAU')
    metricas(history_EXCEPTION, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','EXCEPTION_RNPLLSBNAU')
    guardar_weights('lugares_singles', 'BGM3','EXCEPTION_RNPLLSBNAU')
    
    pred = parallel_model.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , True , 'EXCEPTION_RNPLLSBNAU') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, True, 'EXCEPTION_RNPLLSBNAU') #el false es para  amazon

else:
    #####################################################################
    #                         BMG BATCH                              #
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
                              class_weight=class_weight
                              )
    archivo = 'D:/Modelos Tensorflow/Conjunto train y test/history_prueba/BGM3'
    
    pred = model_BGM3.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM3_ma2_ran') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM3_ma2_ran') #el false es para  amazon
    
    history_dict = history_BGM3.history
    # Save it under the form of a json file
    json.dump(history_dict, open(archivo, 'w'))
    #####################################################################
    #                         BMG3 MAXPOOL                              #
    #####################################################################
    rgb = 3
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM3 = BGM_3(vector_shape, n_categorias)
    model_BGM3.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    history_BGM3 = model_BGM3.fit(x_train_img_z, y_train_z,
                              batch_size = 20,
                              epochs=2,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    archivo = 'D:/Modelos Tensorflow/Conjunto train y test/history_prueba/BGM3'
    
    pred = model_BGM3.predict(x_test_img_z, batch_size=40, verbose=1)
    id_top, proba_top = guardar_top10_str(pred, y_test_z, idimagenes_test_z, n_categorias , False , 'BGM3_ma2_ran') #el false es para  amazon
    guardar_top10_images(id_top,proba_top, False, 'BGM3_ma2_ran') #el false es para  amazon
    
    history_dict = history_BGM3.history
    # Save it under the form of a json file
    json.dump(history_dict, open(archivo, 'w'))
    
    #####################################################################
    #                         BMG3 AVGPOOL                              #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM3 = BGM_3(vector_shape, n_categorias)
    model_BGM3.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    history_BGM3 = model_BGM3.fit(x_train_img_z, y_train_z,
                              batch_size = 20,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    #####################################################################
    #                            DENSENET121                            #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_DENSENET121 = DENSENET121(vector_shape, n_categorias)
    model_DENSENET121.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    
    history_DENSENET121 = model_DENSENET121.fit(x_train_img_z, y_train_z,
                              batch_size = 2,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    
    #####################################################################
    #                            RESNET50                               #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_RESNET50 = RESNET50(vector_shape, n_categorias)
    model_RESNET50.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    
    history_RESNET50 = model_RESNET50.fit(x_train_img_z, y_train_z,
                              batch_size = 2,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    #####################################################################
    #                           INCEPTIONV3                             #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_INCEPTIONV3 = INCEPTIONV3(vector_shape, n_categorias)
    model_INCEPTIONV3.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    
    history_INCEPTIONV3 = model_INCEPTIONV3.fit(x_train_img_z, y_train_z,
                              batch_size = 2,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
    #####################################################################
    #                            EXCEPTION                              #
    #####################################################################
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_EXCEPTION = EXCEPTION(vector_shape, n_categorias)
    model_EXCEPTION.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    history_EXCEPTION = model_EXCEPTION.fit(x_train_img_z, y_train_z,
                              batch_size = 2,
                              epochs=10,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              class_weight=class_weight
                              )
  


##########################################################################################################################################################################################
'''                                      MODELO BGM3_pano                                 '''
##########################################################################################################################################################################################
x_train_img_z, idimagenes_train_z, y_train_z, x_test_img_z, idimagenes_test_z, y_test_z = cargar_imagenes_pano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar', prueba = False, amazon = True, contraste = True)

amazon = True
rgb = 3
if amazon == True:
    #####################################################################
    #                         BGM4 AVGPOOL                             #
    #####################################################################
    img_rows, img_cols = 224, 224*4
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM4 = BGM_4(vector_shape)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM4, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    
    history_BGM4 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 25,
                              epochs=50,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              )
    
    graficos(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM3_ran_pano')
    guardar_history(history_BGM4, 'lugares_singles', 'BGM3','BGM3_ran_pano')
    metricas(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM3_ran_pano')
    guardar_weights('lugares_singles', 'BGM3','BGM3_ran_pano')
    #####################################################################
    #                         BGM4 AVGPOOL                              #
    #####################################################################
    img_rows, img_cols = 224, 224*4
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM4 = BGM_4(vector_shape)
    
    parallel_model = keras.utils.multi_gpu_model(model_BGM4, gpus=2, cpu_merge=True, cpu_relocation=False)
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    
    history_BGM4 = parallel_model.fit(x_train_img_z, y_train_z,
                              batch_size = 25,
                              epochs=15,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              )
    
    graficos(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM4_ran_pano')
    guardar_history(history_BGM4, 'lugares_singles', 'BGM3','BGM4_ran_pano')
    metricas(history_BGM4, x_test_img_z, y_test_z, 'lugares_singles', 'BGM3','BGM4_ran_pano')
    guardar_weights('lugares_singles', 'BGM3','BGM4_ran_pano')
else:
    img_rows, img_cols = 224, 224
    vector_shape = (img_rows,img_cols, rgb)
    model_input = Input(shape=(img_rows,img_cols,rgb))
    model_BGM3 = BGM_3(vector_shape)
    model_BGM3.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    history_BGM3 = model_BGM3.fit(x_train_img_z, y_train_z,
                              batch_size = 20,
                              epochs=15,
                              verbose=1,
                              shuffle = True,
                              validation_data=(x_test_img_z, y_test_z),
                              )







