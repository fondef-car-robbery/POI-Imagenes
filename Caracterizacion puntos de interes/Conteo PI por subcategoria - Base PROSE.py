# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:03:46 2018

@author: Bgm9
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import geopy.distance
import math
import time

#############################################################################################
'''                                     CARGA DE BASES - ROBO                             '''
#############################################################################################
def correccion_subcategoria(df):
    df['subcategoria'] = [str(x).replace(",","") for x in df['subcategoria']]
    df['subcategoria'] = [str(x).replace("-"," ") for x in df['subcategoria']]
    return df

#df_randomsantiago = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Random de Santiago.csv', header=0, encoding = 'latin1')

def df_filtrar_por_tipovehiculo():
    df = pd.read_csv('file:///C:/BGM/robos_prose.csv', header=0, encoding = 'latin1')
    vehiculos_estudiados = ['AUTOMOVIL','STATION WAGON', 'TODO TERRENO','CAMIONETA']
    df = df.loc[df['tve_desc'].isin(vehiculos_estudiados)]
    ids_confiltro_tipoauto = df['id_prose'].to_frame()
    return ids_confiltro_tipoauto

def eliminar_direcciones_repetidas():
    df = pd.read_csv('file:///C:/BGM/DireccionesPROSE/downloadimages.csv', header=0, encoding = 'latin1')
    df = df[np.isfinite(df['latitude'])]
    df1 = df.drop_duplicates(subset=['latitude','longitude'])
    return df1

def ID_Santiago_conimagenes():
    label_directory = 'D:/Modelos Tensorflow/Capturas'   
    lista_idimagenes = os.listdir(label_directory)[1:]
    data = OrderedDict([ 
          ('id_prose', lista_idimagenes)
          ])
    df = pd.DataFrame.from_dict(data)
    df.id_prose = df.id_prose.astype(float)
    df.id_prose = df.id_prose + 1 # Corregimos las ids sumando 1 porque las imagenes parten de 0
    return df
    
df_tipovehiculo = df_filtrar_por_tipovehiculo()
df_direcciones_unica = eliminar_direcciones_repetidas()
df_santiago_imagenes =  ID_Santiago_conimagenes()

df_randomsantiago = pd.merge(df_tipovehiculo, df_direcciones_unica, on='id_prose', how='inner')
df_randomsantiago = pd.merge(df_randomsantiago, df_santiago_imagenes, on='id_prose', how='inner')


df_ambiente =  pd.read_csv('file:///C:/BGM/DireccionesCIVICO/data tipo local/puntosinteres.csv', header=0, encoding = 'utf-8', sep=';')
df_geo_ambiente =  pd.read_csv('file:///C:/BGM/DireccionesCIVICO/example.csv', header=0, encoding = 'latin1')

df_geo_ambiente = df_geo_ambiente.drop_duplicates(subset='direccion', keep='first', inplace=False)

df_ambiente = df_ambiente.rename(columns={'Direccion':'direccion'})
df_ambiente['direccion'] = df_ambiente['direccion'] + ', Santiago, Region Metropolitana, Republica de Chile' 

df_geoamb = pd.merge(df_ambiente, df_geo_ambiente, on='direccion', how='outer')
df_geoamb = df_geoamb.drop_duplicates(subset='Nombre', keep='first', inplace=False)

df_geoamb['longitude'] = pd.to_numeric(df_geoamb['longitude'], downcast='float', errors='coerce')
df_geoamb['latitude'] = pd.to_numeric(df_geoamb['latitude'], downcast='float', errors='coerce')
df_geoamb['longitude'] = df_geoamb['longitude'].astype('float')
df_geoamb = df_geoamb[np.isfinite(df_geoamb['longitude'])]
df_geoamb = correccion_subcategoria(df_geoamb)
df_geoamb = df_geoamb[['subcategoria','latitude','longitude']]
del df_ambiente
del df_geo_ambiente
## Correspondes a los valores que si tienen imagenes
#norobo_oriente = os.listdir('D:\Modelos Tensorflow\prototipo norobo')

    
    
#categorias = list(set(df_geoamb['categoria'])) 
subcategorias = sorted(list(set(df_geoamb['subcategoria'])))
len(subcategorias)

#############################################################################################
'''                       DETECCION DE ELEMENTOS DEL ENTORNO - ROBO                       '''
#############################################################################################

def detectar_subcategoria(var_subcategoria,df_geoamb):
    subcategorias = sorted(list(set(df_geoamb['subcategoria'])))
    lista = [0] * len(subcategorias)    
    for i in range(0,len(subcategorias)):
        if var_subcategoria == subcategorias[i]:
            lista[i] = lista[i] + 1
        else:
            pass
    return lista

def agregar_row_matriz(i,matriz,var):
    c = 0
    for segmento in var:
        r =0
        for d in segmento:
            matriz[i][c+r] = d
            r = r + 1
        c = c + len(segmento)
    return matriz

def lista_imagenes_existentes():
    directorio = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes'
    lsitaimagenes_sin_fitrar = os.listdir(directorio)
    imagenes_existentes = []
    for i in lsitaimagenes_sin_fitrar:
        imagen = directorio+'/' + i + '/0/gsv_0.jpg'
        check = os.path.isfile(imagen)
        if check == True:
            imagenes_existentes.append(i)
    return imagenes_existentes

   
def contador_puntosinteres(base_por_caracterizar,subcategorias,df_geoamb,val_inicial,val_final):
    # Agregar comprobación de que si está en la base entonces no ha que vovler a agregar a ese punto
    df_randomsantiago = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Random de Santiago.csv', header=0, encoding = 'latin1')
    base_por_caracterizar = df_randomsantiago
    #val_inicial = 1
    #val_final = 1
    archivo = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_random_Santiago.csv'
    matriz = ([0]*(len(subcategorias) * 5 +1)) 
    p = 0    
    
    id_con_imagenes = lista_imagenes_existentes()
    id_con_imagenes = OrderedDict([ 
                    ('id', id_con_imagenes),
              ])
    id_con_imagenes = pd.DataFrame.from_dict(id_con_imagenes)
    
    base_por_caracterizar['id'] = base_por_caracterizar['id']-1
    base_por_caracterizar['id'] = base_por_caracterizar['id'].astype('str')
    base_por_caracterizar = pd.merge(base_por_caracterizar, id_con_imagenes, on='id', how='inner')

    base_caracterizada_parcial =  pd.read_csv(archivo, header=0, encoding = 'latin1', usecols=["id","uniformes 45km"])
    base_caracterizada_parcial['id'] = base_caracterizada_parcial['id'].astype('float')
    base_caracterizada_parcial['id'] = base_caracterizada_parcial['id'].astype(str).replace('\.0', '', regex=True)

    base_por_caracterizar = pd.merge(base_por_caracterizar, base_caracterizada_parcial, on='id', how='left')
    
    base_por_caracterizar = base_por_caracterizar[(base_por_caracterizar['uniformes 45km'].isnull() == True)]
    base_por_caracterizar = base_por_caracterizar.loc[:,['id','x','y']]
    
    for i in range(int(len(base_por_caracterizar)*val_inicial),int(len(base_por_caracterizar)*val_final)):
        #base_caracterizada_parcial = list(base_caracterizada_parcial['id'])
        start = time.time()
        id_data = int(base_por_caracterizar.iloc[i]['id']) - 1
        #if id_data not in base_caracterizada_parcial and str(id_data) in id_con_imagenes: 
        xr = base_por_caracterizar.iloc[i]['x']
        yr = base_por_caracterizar.iloc[i]['y']
        r = np.array([xr, yr])
        var_01 = [0] * len(subcategorias)   
        var_12 = [0] * len(subcategorias)   
        var_23 = [0] * len(subcategorias)   
        var_34 = [0] * len(subcategorias)   
        var_45 = [0] * len(subcategorias)     
        for j in range(0,len(df_geoamb)):
            try:
                xa = df_geoamb.iloc[j]['latitude']
                ya = df_geoamb.iloc[j]['longitude']
                subcategoria = df_geoamb.iloc[j]['subcategoria']
                a = np.array([xa, ya])
                distance = round(geopy.distance.vincenty(r, a).km, 4)
                if 5<distance:
                    pass
                elif 0<distance and distance<=1 :w
                    lists_of_lists = [var_01, detectar_subcategoria(subcategoria,df_geoamb)]
                    var_01 = [sum(x) for x in zip(*lists_of_lists)]
                elif 1<distance and distance<=2:
                    lists_of_lists = [var_12, detectar_subcategoria(subcategoria,df_geoamb)]
                    var_12 = [sum(x) for x in zip(*lists_of_lists)]
                elif 2<distance and distance<=3:
                    lists_of_lists = [var_23, detectar_subcategoria(subcategoria,df_geoamb)]
                    var_23 = [sum(x) for x in zip(*lists_of_lists)]
                elif 3<distance and distance<=4:
                    lists_of_lists = [var_34, detectar_subcategoria(subcategoria,df_geoamb)]
                    var_34 = [sum(x) for x in zip(*lists_of_lists)]
                elif 4<distance and distance<=5:
                    lists_of_lists = [var_45, detectar_subcategoria(subcategoria,df_geoamb)]
                    var_45 = [sum(x) for x in zip(*lists_of_lists)]  
            except:
                    pass
        var = [var_01, var_12, var_23,var_34, var_45]
        matriz[0] = id_data
        contador = 1
        for x in var:
            for val in x:
                matriz[contador] = val
                contador = contador + 1

        data = pd.DataFrame.from_dict(matriz).T
        file = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_random_Santiago.csv'
        data.to_csv(file, index=False, mode='a', header=False, encoding='latin1')

        p = p + 1
        stop = time.time()
        duration = stop-start
        val = 'Desarrollo:  ' + str(round((i-int(len(base_por_caracterizar)*val_inicial))/(int(len(base_por_caracterizar)*val_final)-int(len(base_por_caracterizar)*int(len(base_por_caracterizar)*val_inicial)))*100,1)) + '%' + '   tiempo:  ' + str(round(duration))
        print(val)
    return matriz

df_randomsantiago = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/PROSE de Santiago - PI.csv', header=0, encoding = 'latin1')
contador_puntosinteres(df_randomsantiago,subcategorias,df_geoamb,0,1) 

