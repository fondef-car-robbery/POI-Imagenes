# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:35:15 2018

@author: Bgm9
"""
# Shared Input Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from haversine import haversine

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
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import os
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import scipy.misc
from skimage import data, io, filters
from PIL import Image
from random import shuffle
from resizeimage import resizeimage
from numpy import array
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance
from keras.layers import Input, Dense, concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalMaxPool2D


#############################################################################################
'''                         FUNCIONES DE CARGA DE DATOS                                   '''
#############################################################################################
####
''' VERIFICAR:          OJO CON EL ORDEN DE LOS IDS, SI NO ESTARÁ MALO EL ETIQUETADO !! '''
####
def cargar_imagenes_nopano(nombre_etiqueta, metodo_traintest, prueba, amazon, contraste):
    if prueba == True:
        len_train = 250
        channel = True
        tamagno_img = 224
        contraste = True
    else:
        len_train = 10000
        channel = True
        tamagno_img = 224
    
    if amazon == True:
        directorio_fotos = 'C:/Users/Administrator/Documents/Imagenes'
        df_ids = pd.read_csv('file:///C:/Users/Administrator/Documents/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
    else:
        directorio_fotos = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes'
        df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
        #nombre_etiqueta = 'label_lugar'

    
    if metodo_traintest == 'random':
        zona_riesgo = df_ids.loc[df_ids[nombre_etiqueta] == 1]
        zona_riesgo_subset = zona_riesgo.sample(n= int(len_train/2*1.2))
        train_1 = zona_riesgo_subset.sample(n= int(len_train/2))
        test_1 = zona_riesgo_subset[~zona_riesgo_subset.isin(train_1)].dropna()
        
        zona_noriesgo = df_ids.loc[df_ids[nombre_etiqueta] == 0]
        zona_noriesgo_subset = zona_noriesgo.sample(n= int(len_train/2*1.2))
        train_0 = zona_noriesgo_subset.sample(n= int(len_train/2))
        test_0 = zona_noriesgo_subset[~zona_noriesgo_subset.isin(train_0)].dropna()
        
        train = [train_1, train_0]
        test = [test_1, test_0]
        train = pd.concat(train)
        test = pd.concat(test)
        print(len(train))
        print(len(test))
        train = train.sort_values('id', ascending=True)
        train = train.reset_index(drop=True)
        test = test.sort_values('id', ascending=True)
        test = test.reset_index(drop=True)
        
    elif metodo_traintest == 'separar':
        #nombre_etiqueta = 'label_lugar'
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
        print('selection')        
        train_completo = df_ids.loc[df_ids['train'] == 1]
        train_1 = train_completo.loc[df_ids[nombre_etiqueta] == 1]
        train_1 = train_1.sample(n= int(len_train/2))
        train_0 = train_completo.loc[df_ids[nombre_etiqueta] == 0]
        train_0 = train_0.sample(n= int(len_train/2))
        train = [train_1, train_0]
        train = pd.concat(train)
        print('set1')
        test_completo = df_ids.loc[df_ids['train'] == 0]
        test_1 = test_completo.loc[df_ids[nombre_etiqueta] == 1]
        test_1 = test_1.sample(n=int(len_train/2*0.2))
        test_0 = test_completo.loc[df_ids[nombre_etiqueta] == 0]
        test_0 = test_0.sample(n=int(len_train/2*0.2))
        test = [test_1, test_0]
        test = pd.concat(test)
        print('set2')
        
        #import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(train['x'],train['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
        plt.scatter(test['x'],test['y'], alpha=0.1, s =1.5, color='b') # Zona indefinida 
        plt.show()  
        
    elif metodo_traintest == 'cuadrantes':
        pass
        
    t = 0
    imag_train = []
    idimagen_train = []
    etiquetas_train = []
    for row in range(0,len(train)):
        #print(train.iloc[row]['id'])
        #row = 60
        #print(int(train.iloc[row]['id']))
        #print(int(train.iloc[row][nombre_etiqueta]))
        try:
            f = str(int(train.iloc[row]['id'])) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(train.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_train.append(img)
            idimagen_train.append(int(train.iloc[row]['id']))
            if train.iloc[row][nombre_etiqueta] == 0:
                etiquetas_train.append(0)
            else:
                etiquetas_train.append(1)
        except:
            pass
        try:
            f = str(int(train.iloc[row]['id'])) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(train.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_train.append(img)
            idimagen_train.append(int(train.iloc[row]['id']))
            if train.iloc[row][nombre_etiqueta] == 0:
                etiquetas_train.append(0)
            else:
                etiquetas_train.append(1)
        except:
            pass
        try:
            f = str(int(train.iloc[row]['id'])) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(train.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_train.append(img)
            idimagen_train.append(int(train.iloc[row]['id']))
            if train.iloc[row][nombre_etiqueta] == 0:
                etiquetas_train.append(0)
            else:
                etiquetas_train.append(1)
        except:
            pass
        try:
            f = str(int(train.iloc[row]['id'])) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(train.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_train.append(img)
            idimagen_train.append(int(train.iloc[row]['id']))
            if train.iloc[row][nombre_etiqueta] == 0:
                etiquetas_train.append(0)
            else:
                etiquetas_train.append(1)
        except:
            pass
        t = t + 1
    imag_train = np.asarray(imag_train, dtype=np.float32)
    imag_train = imag_train/255
    print(etiquetas_train)
    df_etiquetas = etiquetas_train
    num_classes = 2
    etiquetas_train = keras.utils.to_categorical(df_etiquetas, num_classes).astype('float64')
    print('train listo')
   
    t = 0
    imag_test = []
    idimagen_test = []
    etiquetas_test = []
    for row in range(0,len(test)):  
        try:
            f = str(int(test.iloc[row]['id'])) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(test.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_test.append(img)
            idimagen_test.append(int(test.iloc[row]['id']))
            if test.iloc[row][nombre_etiqueta] == 0:
                etiquetas_test.append(0)
            else:
                etiquetas_test.append(1)
        except:
            pass
        try:
            f = str(int(test.iloc[row]['id'])) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(test.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_test.append(img)
            idimagen_test.append(int(test.iloc[row]['id']))
            if test.iloc[row][nombre_etiqueta] == 0:
                etiquetas_test.append(0)
            else:
                etiquetas_test.append(1)
        except:
            pass
        try:
            f = str(int(test.iloc[row]['id'])) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(test.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_test.append(img)
            idimagen_test.append(int(test.iloc[row]['id']))
            if test.iloc[row][nombre_etiqueta] == 0:
                etiquetas_test.append(0)
            else:
                etiquetas_test.append(1)
        except:
            pass
        try:
            f = str(int(test.iloc[row]['id'])) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = Image.open(directorio_archivo)
            if channel == False:
                #img = img.convert('LA')
                if int(test.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(2000)
            r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_test.append(img)
            idimagen_test.append(int(test.iloc[row]['id']))
            if test.iloc[row][nombre_etiqueta] == 0:
                etiquetas_test.append(0)
            else:
                etiquetas_test.append(1)
        except:
            pass
        t = t + 1
    imag_test = np.asarray(imag_test, dtype=np.float32)
    imag_test = imag_test/255
    print(etiquetas_test)
    df_etiquetas = etiquetas_test
    num_classes = 2
    etiquetas_test = keras.utils.to_categorical(df_etiquetas, num_classes).astype('float64')
    print('test listo')
    return imag_train, idimagen_train, etiquetas_train, imag_test, idimagen_test, etiquetas_test 

def calcular_harvesine(df_PI):
    #df_PI = df_trainPI
    df_PI['dst_plaza'] = 0.0
    df_PI['dst_costanera'] = 0.0 
    for i in range(0,len(df_PI)):
            xp = df_PI.iloc[i]['x']
            yp = df_PI.iloc[i]['y']
            #xp = -33.42218234	
            #yp = -70.61050301
            xplaza = -33.437637
            yplaza = -70.650471
            xcostanera = -33.417313
            ycostanera = -70.606261
            punto_santiago = (xp, yp) 
            plaza_armas = (xplaza, yplaza)
            costanera = (xcostanera, ycostanera)
            
            dst_aplaza = round(haversine(plaza_armas, punto_santiago),3)
            dst_acostanera = round(haversine(costanera, punto_santiago),3)
            
            df_PI.at[i, 'dst_plaza'] = dst_aplaza
            df_PI.at[i, 'dst_costanera'] = dst_acostanera
    return df_PI 
        

def data_PI(idimagen_train, idimagen_test, amazon):
    if amazon == True:
        df_PI = pd.read_csv('file:///C:/Users/Administrator/Downloads/etapa1_PI.csv', header=0, encoding = 'latin1')
    else:
        df_PI = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header=0, encoding = 'latin1')
    
    df_PI = df_PI[~df_PI.duplicated(subset=['id'], keep=False)]
    
    data_train = OrderedDict([('id', idimagen_train)])
    df_idtrain = pd.DataFrame.from_dict(data_train)
    df_trainPI = pd.merge(df_PI, df_idtrain, on ='id', how='right')
    df_trainPI = df_trainPI.fillna(0)
    df_trainPI = calcular_harvesine(df_trainPI)
    
    data_test = OrderedDict([('id', idimagen_test)])
    df_idtest = pd.DataFrame.from_dict(data_test)
    df_testPI = pd.merge(df_PI, df_idtest, on ='id', how='right')
    df_testPI = df_testPI.fillna(0)
    df_testPI = calcular_harvesine(df_testPI)
    
    df_trainPI = df_trainPI.drop(['x','y'], axis=1)
    df_testPI = df_testPI.drop(['x','y'], axis=1)
    
    df_trainPI = df_trainPI.sort_values('id', ascending=True)
    df_trainPI = df_trainPI.reset_index(drop=True)
    df_testPI = df_testPI.sort_values('id', ascending=True)
    df_testPI = df_testPI.reset_index(drop=True)
    
    ''' NORMALIZAR POR EL MAXIMO DE LA BASE'''
    max_PI = 0
    for i in range(1,(len(df_trainPI.columns)-2)):
        if max(df_trainPI.iloc[:,i]) > max_PI:
            max_PI = max(df_trainPI.iloc[:,i])
            
    for i in range(1,(len(df_testPI.columns)-2)):
        if max(df_testPI.iloc[:,i]) > max_PI:
            max_PI = max(df_testPI.iloc[:,i])
            
    for i in range(1,(len(df_testPI.columns)-2)):
        df_trainPI.iloc[:,i] = df_trainPI.iloc[:,i]/max_PI
        df_testPI.iloc[:,i] = df_testPI.iloc[:,i]/max_PI
    
    lista = ['dist_costanera ',
        'servicios.personales.12km', 
        'para.verse.bien.12km', 
        'servicios.personales.01km', 
        'internacional.12km', 
        'dist_plazarmas',
        'cafes.slash.cafeterias.12km', 
        'comunicaciones.12km', 
        'parques.y.plazas.12km', 
        'academias.y.cursos.12km', 
        'cafes.slash.cafeterias.01km', 
        'educativos.12km', 'educativos.01km', 
        'comunicaciones.01km', 
        'concesionarios.y.automotoras.12km', 
        'hospitales.clinicas.centros.de.salud.12km', 
        'deco.hogar.12km', 
        'otros.servicios.12km', 
        'construccion.01km', 
        'transantiago.12km', 
        'comidas.rapidas.12km', 
        'regalos.12km', 
        'hospitales.clinicas.centros.de.salud.01km', 
        'construccion.12km', 
        'para.el.auto.slash.moto.12km', 
        'agricola.12km', 'alimentos.12km', 
        'otros.servicios.01km', 
        'deco.hogar.01km', 'para.ninos.y.bebes.12km', 
        'comercios.especializados.12km', 
        'internacional.01km', 'para.el.auto.slash.moto.01km', 
        'para.verse.bien.01km', 
        'mascotas.12km', 
        'parques.y.plazas.01km', 
        'otros.comercios.12km', 
        'vivienda.12km', 
        'otros.restaurantes.12km', 
        'compras.personales.12km', 
        'servicios.especializados.12km', 
        'alimentos.01km', 'chilena.12km', 
        'comidas.rapidas.01km', 
        'compras.personales.01km', 
        'joyas.y.accesorios.12km', 
        'para.vestirse.12km', 
        'hospedaje.12km', 
        'regalos.01km', 
        'hogar.12km', 
        'bares.y.discotheque.12km', 
        'para.el.hogar.12km', 
        'concesionarios.y.automotoras.01km', 
        'tecnologia.12km', 
        'joyas.y.accesorios.01km', 
        'mascotas.01km', 
        'centros.de.pilates.slash.yoga.12km', 
        'hospedaje.01km', 'terapias.naturales.12km', 
        'cultura.12km', 'vida.social.12km', 
        'transantiago.01km', 
        'empresas.slash.oficinas.12km', 
        'chilena.01km', 'otros.restaurantes.01km', 
        'academias.y.cursos.01km', 'tecnologia.01km', 
        'spa.12km', 'servicio.tecnico.12km', 'spa.01km', 
        'helados.slash.postres.12km', 'zapateria.12km', 
        'otras.compras.12km', 
        'agricola.01km', 
        'para.vestirse.01km', 
        'financieros.12km', 
        'otros.organismos.ciudad.12km', 
        'uniformes.12km', 
        'medicos.y.especialistas.12km', 
        'plasticos.12km', 
        'financieros.01km', 
        'bares.y.discotheque.01km', 
        'otros.comercios.01km', 
        'para.ninos.y.bebes.01km', 
        'cine.12km', 
        'reciclaje.slash.chatarra.12km', 
        'aseo.y.limpieza.01km', 
        'tiendas.electronicas.12km', 
        'electrodomesticos.12km', 
        'urbano.12km', 
        'vegetariano.slash.naturista.12km', 
        'reciclaje.slash.chatarra.01km',
        'vegetariano.slash.naturista.01km', 
        'para.el.hogar.01km', 
        'fundaciones.12km',
        'prevision.de.salud.12km',
        'aseo.y.limpieza.12km',
        'estacionamientos.12km',
        'para.deportistas.12km',
        'centros.comerciales.slash.mall.12km',
        'servicios.especializados.01km', 
        'agencias.de.turismo.12km', 
        'terapias.naturales.01km', 'hogar.01km',
        'embajadas.12km', 'otros.organismos.ciudad.01km', 
        'vivienda.01km', 'servicios.contables.12km',
        'agencias.de.turismo.01km', 
        'parrilladas.12km', 
        'delivery.a.domicilios.12km', 
        'picadas.12km', 'empresas.slash.oficinas.01km', 
        'loteria.12km', 'vida.social.01km',
        'comercios.especializados.01km',
        'urbano.01km', 'otras.compras.01km',
        'helados.slash.postres.01km',
        'plasticos.01km', 'cultura.01km',
        'pescados.y.mariscos.12km',
        'uniformes.01km', 
        'ong.12km',
        'servicios.contables.01km', 
        'centros.comerciales.slash.mall.01km']
    
    #df_trainPI = df_trainPI[df_trainPI.columns & lista]
    
    return df_trainPI, df_testPI

def auc_pr(y_true, y_pred, curve='PR'):
    return tf.metrics.auc(y_true, y_pred, curve=curve)

imag_train, idimagen_train, etiquetas_train, imag_test, idimagen_test, etiquetas_test  = cargar_imagenes_nopano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar', prueba = False, amazon = True, contraste = True)
df_trainPI, df_testPI = data_PI(idimagen_train, idimagen_test, amazon = True)    

df_trainPI = df_trainPI.values.astype('float32')
df_testPI = df_testPI.values.astype('float32')
len(df_trainPI)
len(imag_train)

#############################################################################################
'''                                   MODELO                                   '''
#############################################################################################
# input layer
# first feature extractor
#dense1 = Dense(64, activation='relu')(input1)
#dense2 = Dense(32, activation='relu')(dense1)
#dense3 = Dense(32, activation='relu')(dense2)
#flat1 = dense3#Flatten()(dense3)

# second feature extractor
input1 = Input(shape=(211,), dtype='float32', name='input1')

input2 = Input(shape=(224, 224, 3), dtype='float32', name='input2')
conv1 = Conv2D(256, kernel_size=2, activation='relu')(input2)
x = BatchNormalization()(conv1)
x = Activation('elu')(x)
pool1 = MaxPooling2D(pool_size=(2, 2))(x)
conv2 = Conv2D(64, kernel_size=2, activation='relu')(pool1)
x = BatchNormalization()(conv2)
x = Activation('elu')(x)
pool2 = MaxPooling2D(pool_size=(2, 2))(x)
conv3 = Conv2D(64, kernel_size=2, activation='relu')(pool2)
x = BatchNormalization()(conv3)
x = Activation('elu')(x)
conv4 = Conv2D(125, kernel_size=2, activation='relu')(x)
conv5 = Conv2D(100, kernel_size=2, activation='relu')(conv4)
conv6 = Conv2D(75, kernel_size=2, activation='relu')(conv5)
#globalavg_pool = GlobalMaxPool2D()(conv6)
flat = Flatten()(conv6)

# merge feature extractors
merge = concatenate([input1, flat])
# interpretation layer
#globalavg_pool = GlobalMaxPool2D()(merge)

#hidden1 = Dense(100, activation='relu')(merge)
#hidden2 = Dense(100, activation='relu')(hidden1)
#hidden3 = Dense(100, activation='relu')(hidden2)
# prediction output
output = Dense(2, activation='sigmoid', dtype='float64')(merge)
#del model
model = Model(inputs=[input1,input2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='shared_input_layer.png')
#print(x_train.shape)
#print(x_train_img.shape)
#print(y_train.shape)
#print(y_train.shape)


#print(y_train.shape)

#train_y.reshape(-1,1)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit( [df_trainPI, imag_train], etiquetas_train,
                      batch_size = 10,
                      epochs=50,
                      verbose=1,
                      shuffle = True)

df_trainPI.shape, imag_train.shape, etiquetas_train.shape
#,
#                      validation_data=(x_test, y_test),
#                      )

score = model.evaluate([x_test, x_test_img],
                        y_test,
                        verbose=1)


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





print('Test loss:', score[0])
print('Test accuracy:', score[1])
###############################################################################
#                           CODIGO DE KAGGLE
###############################################################################
Input_figure = Input(shape=(224,224,3), name='input1')
Input_angle = Input(shape=(211,), name = 'input2')

y = Dense(500)(Input_angle)
y = BatchNormalization()(y)
y = Activation('elu')(y)
y = Dropout(0.3)(y)
y = Dense(500)(Input_angle)
y = BatchNormalization()(y)
y = Activation('elu')(y)
y = Dropout(0.3)(y)

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

#concatenate x and angle 
x = concatenate([x, y])

x = Dense(500)(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = Dropout(0.3)(x)
x = Dense(500, activation='elu')(x)
out = Dense(2, activation='sigmoid')(x)

model = Model(inputs=[Input_figure, Input_angle], outputs=out)

###############################################################################
#                           CODIGO DE DATAFUSION
###############################################################################
Input_figure = Input(shape=(224,224,3), name='input1')
Input_angle = Input(shape=(211,), name = 'input2')

y = Dense(500)(Input_angle)
y = BatchNormalization()(y)
y = Activation('elu')(y)
y = Dropout(0.3)(y)
y = Dense(500)(Input_angle)
y = BatchNormalization()(y)
y = Activation('elu')(y)
y = Dropout(0.3)(y)

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

#concatenate x and angle 
x = concatenate([x, y])

x = Dense(500)(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = Dropout(0.3)(x)
x = Dense(500, activation='elu')(x)
out = Dense(2, activation='sigmoid')(x)

model = Model(inputs=[Input_figure, Input_angle], outputs=out)

###############################################################################
#                           CORRER MODELO
###############################################################################
imag_train, idimagen_train, etiquetas_train, imag_test, idimagen_test, etiquetas_test  = cargar_imagenes_nopano(metodo_traintest = 'random',nombre_etiqueta = 'label_lugar', prueba = False, amazon = True, contraste = True)
df_trainPI, df_testPI = data_PI(idimagen_train, idimagen_test, amazon = True)    

df_trainPI = df_trainPI.values.astype('float32')
df_testPI = df_testPI.values.astype('float32')
len(df_trainPI)
len(imag_train)
from sklearn.metrics import classification_report

#amazon = False
#if amazon == True:
opt = keras.optimizers.nadam()
parallel_model = keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=True, cpu_relocation=False)
parallel_model.compile(loss='binary_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'])

history_datafusion = parallel_model.fit([imag_train, df_trainPI], etiquetas_train,
                          batch_size = 60,
                          epochs=30,
                          verbose=1,
                          shuffle = True,
                          validation_data=([imag_test, df_testPI], etiquetas_test),
                          )

graficos(history_datafusion, [imag_test, df_testPI], etiquetas_test, 'lugares_singles', 'BGM3','Datafusion_v3_30ep_r17')
guardar_history(history_datafusion, 'lugares_singles', 'BGM3','Datafusion_v3_30ep_r17')
metricas(history_datafusion, [imag_test, df_testPI], etiquetas_test, 'lugares_singles', 'BGM3','Datafusion_v3_30ep_r17')
guardar_weights('lugares_singles', 'BGM3','Datafusion_v3_30ep_r17')
ultimo = 'Datafusion_v3_30ep_r17◄'
print(ultimo)







