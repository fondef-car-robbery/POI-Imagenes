# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:12:10 2018

@author: Bgm9
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import os

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)

def df_filtrar_por_tipovehiculo():
    df = pd.read_csv('file:///C:/BGM/robos_prose.csv', header=0, encoding = 'latin1')
    vehiculos_estudiados = ['AUTOMOVIL','STATION WAGON', 'TODO TERRENO','CAMIONETA']
    df = df.loc[df['tve_desc'].isin(vehiculos_estudiados)]
    ids_confiltro_tipoauto = df['id_prose']
    return ids_confiltro_tipoauto

def eliminar_direcciones_repetidas():
    df = pd.read_csv('file:///C:/BGM/DireccionesPROSE/downloadimages.csv', header=0, encoding = 'latin1')
    df = df[np.isfinite(df['latitude'])]
    df1 = df.drop_duplicates(subset=['latitude','longitude'])
    return df1

def eliminar_direcciones_sinnumeracion():
    # Tiene la brecha de mejora de eliminar unos valores que tienen un 0 
    df = pd.read_csv('file:///C:/BGM/robos_prose.csv', header=0, encoding = 'latin1')
    df['direc_exact'] = False
    for i in range(0,len(df)):
        try:
            if hasNumbers(df.iloc[i]['sin_direccion_siniestro']) == True:
                df.at[i, 'direc_exact'] = True
        except:
            pass
    df = df.loc[df['direc_exact'] == True]
    ids_direc_exacta = df['id_prose'].to_frame(name='id_prose')
    return ids_direc_exacta

def eliminar_denuncias_erroneas():
    df = pd.read_csv('file:///C:/BGM/robos_prose.csv', header=0, encoding = 'latin1')
    df = df.loc[df['obs_ult_estado'].isnull() == True]
    ids_denuncias_noerroneas = df['id_prose'].to_frame(name='id_prose')
    return ids_denuncias_noerroneas

def ID_Santiago_conimagenes():
    label_directory = 'D:/Modelos Tensorflow/Capturas'   
    lista_idimagenes = os.listdir(label_directory)
    data = OrderedDict([ 
          ('id_prose', lista_idimagenes)
          ])
    df = pd.DataFrame.from_dict(data)
    return df

################################################################################################
#              '''    PENDIENTE: Implementar algún filtro de tipo de delito    '''  
################################################################################################
    
