# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:04:37 2018

@author: Bgm9
"""
import numpy as np
import os
import pandas as pd
from shutil import rmtree
from PIL import Image


#####################################################################################################
'''                       VOLVER A DESCARGAR LAS IMAGENES CON DAÑO                                ''' # Esto corre solo en ambiente Tensorflow
#####################################################################################################

    
label_directory = 'E:/MEMORIA ROBOS/Modelos Tensorflow/Imagenes/'   
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
        