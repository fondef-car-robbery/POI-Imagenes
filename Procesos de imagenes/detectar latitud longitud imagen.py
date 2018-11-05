# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:44:22 2018

@author: Bgm9
"""

import pandas as pd
import os
import random

import seaborn as sns

import matplotlib.pyplot as plt

# library
import seaborn as sns
import pandas as pd
import numpy as np
import google_streetview.api
import os
import sys
import geopy
import csv
import googlemaps
import pandas as pd
import numpy as np
import re
import json
from collections import OrderedDict


def lista_imagenes_existentes():
    #directorio = 'D:/Modelos Tensorflow/Capturas'
    directorio = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes'
    # para ver el caso de imagenes cambiar el directorio
    lsitaimagenes_sin_fitrar = os.listdir(directorio)
    imagenes_existentes = []
    metada_imagenes = []
    for i in lsitaimagenes_sin_fitrar:
        #imagen = directorio+'/' + i + '/0/gsv_0.jpg'
        metadata = directorio+'/' + i + '/0/metadata.json'
        check = True#os.path.isfile(imagen)
        if check == True:
            imagenes_existentes.append(i)
            metada_imagenes.append(metadata)
    return imagenes_existentes, metada_imagenes

imagenes_existentes, metadata_directorios = lista_imagenes_existentes()

ids, xs, ys, fecha = [], [], [], []
errores = []
for i in range(0,len(imagenes_existentes)):
    val = str(i/209192*100)
    print(val)
    archivo = metadata_directorios[i]
    idim = imagenes_existentes[i]
    try:
        with open(archivo) as f:
            data = json.load(f)
            x = data[0]["location"]["lat"]
            y = data[0]["location"]["lng"]
            date = data[0]["date"]
            ids.append(idim)
            xs.append(x)
            ys.append(y)
            fecha.append(date)
    except:
        errores.append(ids)
        
data = OrderedDict([ 
                    ('id', ids),
                    ('x', xs),
                    ('y', ys),
                    ('fecha', fecha)
              ])
            
data = pd.DataFrame.from_dict(data)
        
def graficar(df):
    plt.figure(figsize=(8, 8))
    plt.scatter(df['x'],df['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
    return plt.show()

data.to_csv('D:/Modelos Tensorflow/Conjunto train y test/etapa0.csv', index=True, mode='w', header=True)

graficar(data)

base_etiquetada = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_final_train_test_XY_etiqueta.csv', header=0, encoding = 'utf-8')


        
        
