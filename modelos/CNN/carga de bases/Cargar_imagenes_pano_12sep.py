# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 20:43:08 2018

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
import cv2
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

def cargar_imagenes_pano(nombre_etiqueta, metodo_traintest, prueba, amazon, balance, contraste, channel, augmentation):
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
            num_row = len(df_ids.loc[df_ids[nombre_etiqueta] == 1])
            len_train = num_row
            
        zona_riesgo = df_ids.loc[df_ids[nombre_etiqueta] == 1]
        zona_riesgo_subset = zona_riesgo.sample(n= int(len_train))
        train_1 = zona_riesgo_subset.sample(n= int(len_train*0.8))
        test_1 = zona_riesgo_subset[~zona_riesgo_subset.isin(train_1)].dropna()
        
        zona_noriesgo = df_ids.loc[df_ids[nombre_etiqueta] == 0]
        zona_noriesgo_subset = zona_noriesgo.sample(n= int(len_train))
        train_0 = zona_noriesgo_subset.sample(n= int(len_train*0.8))
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
    imag_train_img1 = []
    imag_train_img2 = []
    imag_train_img3 = []
    imag_train_img4 = []
    
    idimagen_train = []
    etiquetas_train = []
    
    if prueba == True:
        train_load = min(100, len(train))
    else:
        train_load = min(15000, len(train))
        
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
            f = str(id_img) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img]
            value = random.choice([True, False, False, False, False, False, False])
            if valor_label == 0:
                img = random_filter(img, value, valor_label)
            #img = array(r)
            img1 = np.asarray(img, dtype=np.float32)

            f = str(id_img) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            value = random.choice([True, False, False, False, False, False, False])
            if valor_label == 0:
                img = random_filter(img, value, valor_label)
            #img = array(r)
            img2 = np.asarray(img, dtype=np.float32)
            

            f = str(id_img) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            value = random.choice([True, False, False, False, False, False, False])
            if valor_label == 0:
                img = random_filter(img, value, valor_label)
            #img = array(r)
            img3 = np.asarray(img, dtype=np.float32)
            
            
            f = str(id_img) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            #r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
            value = random.choice([True, False, False, False, False, False, False])
            if valor_label == 0:
                img = random_filter(img, value, valor_label)
            #img = array(r)
            img4 = np.asarray(img, dtype=np.float32)
            
            if img1 != None and img2 != None and img3 != None and img4 != None:
                imag_train_img1.append(img1)
                imag_train_img2.append(img2)
                imag_train_img3.append(img3)
                imag_train_img4.append(img4)
                idimagen_train.append(id_img)
                etiquetas_train.append(valor_label)
        except:
            pass
        t = t + 1
    imag_train_img1 = np.asarray(imag_train_img1, dtype=np.float32)
    imag_train_img2 = np.asarray(imag_train_img2, dtype=np.float32)
    imag_train_img3 = np.asarray(imag_train_img3, dtype=np.float32)
    imag_train_img4 = np.asarray(imag_train_img4, dtype=np.float32)
    imag_train_img1 = imag_train_img1/255
    imag_train_img2 = imag_train_img2/255
    imag_train_img3 = imag_train_img3/255
    imag_train_img4 = imag_train_img4/255
    
    print(etiquetas_train)
    df_etiquetas = etiquetas_train
    etiquetas_train = keras.utils.to_categorical(df_etiquetas, n_categorias).astype('int32')
    print('train listo')
   
    t = 0
    imag_test_img1 = []
    imag_test_img2 = []
    imag_test_img3 = []
    imag_test_img4 = []
    
    idimagen_test = []
    etiquetas_test = []
    if prueba == True:
        test_load = min(100, len(test))
    else:
        test_load = min(5000, len(test))
        
    for row in range(0,test_load):  
        id_img = int(test.iloc[row]['id'])
        valor_label = test.iloc[row][nombre_etiqueta]
        valor_label = asignar_label(valor_label, nombre_etiqueta)
        #row = 8
        try:
            f = str(id_img) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            img1 = np.asarray(img, dtype=np.float32)
            

            f = str(id_img) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            img2 = np.asarray(img, dtype=np.float32)
            
            
            f = str(id_img) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            img3 = np.asarray(img, dtype=np.float32)
            
            
            f = str(id_img) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            if amazon == False or np.shape(img[0])[0] != 224:
                img = cv2.resize(img, (224, 224)) 
            img4 = np.asarray(img, dtype=np.float32)
            if img1 != None and img2 != None and img3 != None and img4 != None:
                imag_test_img1.append(img1)
                imag_test_img2.append(img2)
                imag_test_img3.append(img3)
                imag_test_img4.append(img4)
                idimagen_test.append(id_img)
                etiquetas_test.append(valor_label)
        except:
            pass
        t = t + 1
    imag_test_img1 = np.asarray(imag_test_img1, dtype=np.float32)
    imag_test_img2 = np.asarray(imag_test_img2, dtype=np.float32)
    imag_test_img3 = np.asarray(imag_test_img3, dtype=np.float32)
    imag_test_img4 = np.asarray(imag_test_img4, dtype=np.float32)
    imag_test_img1 = imag_test_img1/255
    imag_test_img2 = imag_test_img2/255
    imag_test_img3 = imag_test_img3/255
    imag_test_img4 = imag_test_img4/255
    
    print(etiquetas_test)
    df_etiquetas = etiquetas_test
    etiquetas_test = keras.utils.to_categorical(df_etiquetas, n_categorias).astype('int32')
    print('test listo')
    return(imag_train_img1, imag_train_img2, imag_train_img3, imag_train_img4, 
           idimagen_train, etiquetas_train, 
           imag_test_img1, imag_test_img2, imag_test_img3, imag_test_img4, 
           idimagen_test, etiquetas_test, 
           n_categorias)