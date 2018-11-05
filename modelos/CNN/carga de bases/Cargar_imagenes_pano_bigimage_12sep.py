# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:02:22 2018

@author: Administrator
"""

# Imports
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
import time

import cv2
from skimage import data, exposure, img_as_float
from skimage import restoration
#AIzaSyA6R_F-GWwZeWHMvioOhKfZgqcAxpmzngY

#############################################################################################
'''                     FUNCIONES DE CARGA IMAGENES NO PANORAMICAS                        '''
#############################################################################################
def asignar_label(valor_label, nombre_etiqueta):
    if nombre_etiqueta == 'label_lugar':
        if valor_label == 0:
            return 0
        elif valor_label > 0:
            return 1
    elif nombre_etiqueta == 'label_lugar>2':
        if valor_label == 0:
            return 0
        elif valor_label == 1:
            return 1
        elif valor_label >= 2:
            return 2
    elif nombre_etiqueta == 'label_f(lugar)':
        if valor_label == 0:
            return 0
        elif valor_label == 1:
            return 1
        elif valor_label == 2:
            return 2
        elif valor_label >= 3:
            return 3
        return valor_label

def num_categorias(nombre_etiqueta, df):
    dataframe_labels =  df[nombre_etiqueta]
    dataframe_labels = dataframe_labels.tolist()
    dataframe_labels = list(set(dataframe_labels))
    conjunto_label = [int(x) for x in dataframe_labels if str(x) != 'nan']
    n_categorias = len(conjunto_label)
    return n_categorias, conjunto_label

def num_categorias_v2(nombre_etiqueta):
    if nombre_etiqueta == 'label_lugar':
        n_categorias = 2
    elif nombre_etiqueta == 'label_lugar>2':
        n_categorias = 3
    elif nombre_etiqueta == 'label_f(lugar)':
        n_categorias = 4
    return n_categorias    

def random_filter(img, value, label):
    if value == True and label == 0:
        num = random.randint(1, 8)
        if num == 1:
            img = cv2.GaussianBlur(img,(7,7),0)
        elif num == 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        elif num == 3:
            img = exposure.adjust_gamma(img, 2)
        elif num == 4:
            img = exposure.adjust_gamma(img, 0.4)
        elif num == 5:
            alpha = float(1.7)  
            beta = int(20)          
            mul_img = cv2.multiply(img, np.array([alpha]))                 
            img = cv2.add(mul_img, beta)                
        elif num == 6:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h += 50 # 4
            s += 0 # 5
            v += 0 # 6
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        elif num == 7:
            img = cv2.flip(img, flipCode=1)
    return img

def data_augmentation(n_categorias, id_train, nombre_etiqueta, amazon):
    #n_categorias = n_categorias
    #id_train = idimagenes_train_z
    #nombre_etiqueta = 'label_f(lugar)'
    #amazon = False
    if amazon == True:
        directorio_save = 'C:/Users/Administrator/Documents/data_augmentation/'
        df_ids = pd.read_csv('file:///C:/Users/Administrator/Documents/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
    else:
        directorio_save = 'C:/data_augmentation/'
        df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
    
    #directorio_save = 'C:/data_augmentation/'
    archivos = os.listdir(directorio_save)

    img_augment = []
    idimagen_augment = []
    etiquetas_augment = []
    for archivo in archivos:
        #archivo = archivos[0]
        p = archivo.find('_')
        id_rec = int(archivo[0:p])
        value = random.choice([1,2,3,4,5,6,7]) # ESto es para no aumentar demasiado la data (elegimos 5 de 7 en esperanza)
        flip_value = archivo.find('f.jpg')
        if value <= 1 or flip_value != -1:
            if id_rec in id_train:
                f = archivo
                directorio_archivo = os.path.join(directorio_save, f)
                img = Image.open(directorio_archivo)
                img = np.asarray(img, dtype=np.float32)
    
                img_augment.append(img)
                idimagen_augment.append(id_rec)
                z = df_ids.loc[df_ids['id'] == id_rec]
                z = z.iloc[0][nombre_etiqueta].astype('int32')
                z = asignar_label(z , nombre_etiqueta)
                etiquetas_augment.append(z) 
            
    img_augment = np.asarray(img_augment, dtype=np.float32)
    img_augment = img_augment/255

    df_etiquetas = etiquetas_augment
    df_etiquetas_augment = keras.utils.to_categorical(df_etiquetas, n_categorias).astype('int32')

    return img_augment, df_etiquetas_augment, idimagen_augment

def cargar_imagenes_bigpano(nombre_etiqueta, metodo_traintest, prueba, amazon, balance, contraste, channel, augmentation):
    # Si Channel es FALSE entonces se aplica contraste sobre las imagenes de no robo
    # Los balance false con metodo train random solo considera la etiqueta de label_lugar 
        
    #tamagno_img = 224
    if amazon == True:
        directorio_fotos = 'C:/Users/Administrator/Documents/Imagenes'
        df_ids = pd.read_csv('file:///C:/Users/Administrator/Documents/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
    else:
        directorio_fotos = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes'
        df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')
        #nombre_etiqueta = 'label_lugar'
        #nombre_etiqueta = 'label_lugar>2'
        #nombre_etiqueta = 'label_f(lugar)'
        #prueba = False
        #prueba = True
        #balance = False
        
    n_categorias, conjunto_label = num_categorias(nombre_etiqueta, df_ids)
    conjunto_label.sort()
    if balance == False and metodo_traintest == 'random':
        #nombre_etiqueta = 'label_f(lugar)'
        conjuntos_train = []
        conjuntos_test = []
        for label in conjunto_label:
            #label = conjunto_label[1]
            if label <= 3:
                datalabel = df_ids.loc[df_ids[nombre_etiqueta] == label]
                if label == 0:
                    datalabel = datalabel.sample(n= 20000) #antes 35800 04 Septiembre
                else:
                    pass
                train = datalabel.sample(n= int(len(datalabel)*0.8))
                test = datalabel[~datalabel.isin(train)].dropna()
                conjuntos_train.append(train)
                conjuntos_test.append(test)
            else:
                datalabel = df_ids.loc[df_ids[nombre_etiqueta] > label]
                train = datalabel.sample(n= int(len(datalabel)*0.8))
                test = datalabel[~datalabel.isin(train)].dropna()
                conjuntos_train.append(train)
                conjuntos_test.append(test)
                break
        train = pd.concat(conjuntos_train)
        test = pd.concat(conjuntos_test)
        
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        #sum(train['label_lugar'])*5
        #sum(test['label_lugar'])
        #sum(train['label_lugar'])/len(train)
        #len(train) - sum(train['label_lugar'])
        
    if balance == True and metodo_traintest == 'random':
        if prueba == True:
            len_train = 300
        else:
            len_train = 8000
            
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
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        print(len(train))
        print(len(test))

    elif balance == True and metodo_traintest == 'separar':
        if prueba == True:
            len_train = 300
        else:
            len_train = 8000
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
        
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        
        #import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(train['x'],train['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
        plt.scatter(test['x'],test['y'], alpha=0.1, s =1.5, color='b') # Zona indefinida 
        plt.show()              
        
    elif balance == False and metodo_traintest == 'separar':
        if prueba == True:
            df_ids = df_ids.sample(n= int((len(df_ids)*0.01)))
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
                
        train = df_ids.loc[df_ids['train'] == 1]
        test = df_ids.loc[df_ids['train'] == 0]
        
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        #import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(train['x'],train['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
        plt.scatter(test['x'],test['y'], alpha=0.1, s =1.5, color='b') # Zona indefinida 
        plt.show()  
        
    elif metodo_traintest == 'cuadrantes':
        pass
        
    n_categorias = num_categorias_v2(nombre_etiqueta)
    t = 0
    imag_train = []
    idimagen_train = []
    etiquetas_train = []
    if prueba == True:
        train_load = 100
    else:
        train_load = 15000
        
    c = 0
    y = 0
    for row in range(0,train_load):
        if c == 1000:
            valor = round(y/train_load, 3)
            print(valor)
            c = 0            
        c = c + 1
        y = y + 1
        #valor_label = train.iloc[row][nombre_etiqueta]
        #valor_label = asignar_label(valor_label, nombre_etiqueta)
        #print(valor_label)    
        #if valor_label == 1:
        #    break
        #row = 60
        #print(int(train.iloc[row]['id']))
        #print(int(train.iloc[row][nombre_etiqueta]))
        id_img = int(train.iloc[row]['id'])
        valor_label = train.iloc[row][nombre_etiqueta]
        valor_label = asignar_label(valor_label, nombre_etiqueta)
        try:
            f = str(id_img) + '/gsv_pano.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224*(4))) 
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img]
            value = random.choice([True, False, False, False, False, False, False])
            if valor_label == 0:
                img = random_filter(img, value, valor_label)
            #img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_train.append(img)
            idimagen_train.append(id_img)
            etiquetas_train.append(valor_label)
        except:
            pass
        t = t + 1
    imag_train = np.asarray(imag_train, dtype=np.float32)
    imag_train = imag_train/255
    print(etiquetas_train)
    df_etiquetas = etiquetas_train
    etiquetas_train = keras.utils.to_categorical(df_etiquetas, n_categorias).astype('int32')
    print('train listo')
   
    t = 0
    imag_test = []
    idimagen_test = []
    etiquetas_test = []
    if prueba == True:
        test_load = 100
    else:
        test_load = 5000
        
    for row in range(0,test_load):  
        id_img = int(test.iloc[row]['id'])
        valor_label = test.iloc[row][nombre_etiqueta]
        valor_label = asignar_label(valor_label, nombre_etiqueta)
        #row = 8
        try:
            f = str(id_img) + '/gsv_pano.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224*(4))) 
            #if channel == False:
                #img = img.convert('LA')
            #    if int(test.iloc[row][nombre_etiqueta]) == 0 and contraste == True:
            #        contrast = ImageEnhance.Contrast(img)
            #        img = contrast.enhance(2000)
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            #img = array(r)
            img = np.asarray(img, dtype=np.float32)
            imag_test.append(img)
            idimagen_test.append(id_img)
            etiquetas_test.append(valor_label)
        except:
            pass
        t = t + 1
    imag_test = np.asarray(imag_test, dtype=np.float32)
    imag_test = imag_test/255
    print(etiquetas_test)
    df_etiquetas = etiquetas_test
    etiquetas_test = keras.utils.to_categorical(df_etiquetas, n_categorias).astype('int32')
    print('test listo')
    return imag_train, idimagen_train, etiquetas_train, imag_test, idimagen_test, etiquetas_test, n_categorias 
