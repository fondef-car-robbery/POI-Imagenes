# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:42:23 2018

@author: Bgm9
"""

import numpy as np
import os
import pandas as pd
from PIL import Image

#####################################################################################################
'''                       GENERACIÓN DE BASE PARA ENTRENAMIENTO CONCATENADA                       ''' # Esto corre solo en ambiente Tensorflow
#####################################################################################################
label_directory = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/'   
lista_idimagenes = os.listdir(label_directory)

# ver si ID pertenece o no a providencia o Santiago 
id_finales = pd.read_csv('C:/BGM/Santiago/id_subset.csv', header=0, encoding = 'latin1') ## Acá estan las id de provi, santiago, etc


var = list(id_finales['id_subdata'])


sinimagenes = []
for j in lista_idimagenes:
    #j = lista_idimagenes[0]
    j = int(j)
    try:
        id_prose = str(j)
        imagen0 = label_directory + id_prose + '/0/gsv_0.jpg'
        imagen90 = label_directory + id_prose + '/90/gsv_0.jpg'
        imagen180 = label_directory + id_prose + '/180/gsv_0.jpg'
        imagen270 = label_directory + id_prose + '/270/gsv_0.jpg'
        
        list_im = [imagen0, 
                   imagen90, 
                   imagen180, 
                   imagen270]
    
        imgs    = [ Image.open(i) for i in list_im ]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here) // vstack for horizontal
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        
        # save that beautiful picture
        imgs_comb = Image.fromarray( imgs_comb)
        guardar_en = label_directory + str(j) + '/' + 'gsv_pano' + '.jpg'
        imgs_comb.save( guardar_en )    
    except:
        sinimagenes.append(id_prose)
