# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:22:41 2018

@author: Bgm9
"""

import numpy as np
import os
import pandas as pd
from shutil import rmtree
from PIL import Image



#####################################################################################################
'''                       ELIMINAR CARPETAS DE 45, 135, 225, 315                                  ''' # Esto corre solo en ambiente Tensorflow
#####################################################################################################
#label_directory = 'E:/MEMORIA ROBOS/Modelos Tensorflow/Capturas/' #disco externo
label_directory = 'D:/Modelos Tensorflow/Capturas/' #disco interno
lista_idimagenes = os.listdir(label_directory)

sinimagenes = []
contador = 0 
for j in lista_idimagenes:
    #j = lista_idimagenes[0]
    j = int(j)
    try:
        id_prose = str(j)
        imagen45 = label_directory + id_prose + '/45'
        imagen135 = label_directory + id_prose + '/135'
        imagen225 = label_directory + id_prose + '/225'
        imagen315 = label_directory + id_prose + '/315'
        rmtree(imagen45)
        rmtree(imagen135)
        rmtree(imagen225)
        rmtree(imagen315)
        contador = contador + 1
        valor = contador/len(lista_idimagenes)*100
        valor = str(valor) + '%'
        print(valor)
    except:
        contador = contador + 1
        valor = contador/len(lista_idimagenes)*100
        valor = str(valor) + '%'
        print(valor)
        
#####################################################################################################
'''                       ELIMINAR CARPETAS SIN IMAGENES                       ''' # Esto corre solo en ambiente Tensorflow
#####################################################################################################
        
#label_directory = 'E:/MEMORIA ROBOS/Modelos Tensorflow/Imagenes/' #disco externo   
label_directory = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/' #disco interno   
lista_idimagenes = os.listdir(label_directory)

sinimagenes = []
contador = 0 
for j in lista_idimagenes:
    #j = lista_idimagenes[0]
    j = int(j)
    try:
        id_prose = str(j)
        imagen0 = label_directory + id_prose + '/0/gsv_0.jpg'
        Image.open(imagen0)
        contador = contador + 1
        valor = contador/len(lista_idimagenes)*100
        valor = str(valor) + '%'
        #print(valor)
        #print('try1')
    except:
        try:
            id_prose = str(j)
            imagen0 = label_directory + id_prose + '/90/gsv_0.jpg'
            Image.open(imagen0)
            contador = contador + 1
            valor = contador/len(lista_idimagenes)*100
            valor = str(valor) + '%'
            #print(valor)
            #print('try2')
        except:
            folder = label_directory + id_prose
            print(folder)
            rmtree(folder)
            valor = contador/len(lista_idimagenes)*100
            valor = str(valor) + '%'
            print(valor)
            print('except2')



