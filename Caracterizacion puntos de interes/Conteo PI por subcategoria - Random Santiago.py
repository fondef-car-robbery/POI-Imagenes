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

#file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_zona_train.csv
#file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_zona_test.csv
#file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_lugar_train.csv
#file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_lugar_test.csv


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

df_geoamb = df_geoamb.sort_values(by=['longitude'], ascending=True)
df_geoamb = df_geoamb.reset_index(drop=True)



#categorias = list(set(df_geoamb['categoria']))
subcategorias = sorted(list(set(df_geoamb['subcategoria'])))
len(subcategorias)

#############################################################################################
'''                       DETECCION DE ELEMENTOS DEL ENTORNO - ROBO                       '''
#############################################################################################

def buscar_rango(df_geoamb, x,y):
    #i=10000
    #y = base_por_caracterizar.iloc[i]['y']
    #x = base_por_caracterizar.iloc[i]['x']

    x_max = x + 0.0180
    x_min = x - 0.0180
    y_max = y + 0.0180
    y_min = y - 0.0180
    subset_geoamb = df_geoamb.loc[(df_geoamb['latitude'] > x_min) & (df_geoamb['latitude'] < x_max)]
    subset_geoamb = subset_geoamb.loc[(subset_geoamb['longitude'] > y_min) & (subset_geoamb['longitude'] < y_max)]
    subset_geoamb = subset_geoamb.reset_index(drop=True)
    #for j in range(0,len(subset_geoamb)):
    #    distance = geopy.distance.vincenty((x, y),(subset_geoamb.iloc[j]['latitude'], subset_geoamb.iloc[j]['longitude'])).km
    #    print(distance)
    return subset_geoamb

def detectar_subcategoria(var_subcategoria,subcategorias):
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
    #base_por_caracterizar = pd.read_csv('D:/Modelos Tensorflow/Conjunto train y test/etapa1.csv', header=0, encoding = 'latin1')
    #val_inicial = 0.0
    #val_final = 0.5

    archivo = 'D:/Modelos Tensorflow/Conjunto train y test/prose_PI.csv'
    p = 0


    #id_con_imagenes = lista_imagenes_existentes()
    #id_con_imagenes = OrderedDict([
    #                ('id', id_con_imagenes),
    #          ])
    #id_con_imagenes = pd.DataFrame.from_dict(id_con_imagenes)

    base_por_caracterizar['id'] = base_por_caracterizar['id']
    base_por_caracterizar['id'] = base_por_caracterizar['id'].astype('str').replace('\.0', '', regex=True)

    #base_por_caracterizar = pd.merge(base_por_caracterizar, id_con_imagenes, on='id', how='inner')

    base_caracterizada_parcial =  pd.read_csv(archivo, header=0, encoding = 'latin1', usecols=["id","uniformes 01km"])
    base_caracterizada_parcial['id'] = base_caracterizada_parcial['id'].astype('float')
    base_caracterizada_parcial['id'] = base_caracterizada_parcial['id'].astype(str).replace('\.0', '', regex=True)

    base_por_caracterizar = pd.merge(base_por_caracterizar, base_caracterizada_parcial, on='id', how='left')

    base_por_caracterizar = base_por_caracterizar[(base_por_caracterizar['uniformes 01km'].isnull() == True)]
    base_por_caracterizar = base_por_caracterizar.loc[:,['id','x','y']]
    subcategorias = sorted(list(set(df_geoamb['subcategoria'])))

    for i in range(int(len(base_por_caracterizar)*val_inicial),int(len(base_por_caracterizar)*val_final)):
        #i = 1000
        start = time.time()
        id_data = int(base_por_caracterizar.iloc[i]['id'])
        #print(i)
        #print(id_data)
        #if id_data not in base_caracterizada_parcial and str(id_data) in id_con_imagenes:
        #xr = base_por_caracterizar.iloc[i]['x']
        #yr = base_por_caracterizar.iloc[i]['y']
        #r = np.array([xr, yr])
        matriz = ([0]*(len(subcategorias) * 2 +3))
        var_01 = [0] * len(subcategorias)
        var_12 = [0] * len(subcategorias)
        #var_23 = [0] * len(subcategorias)
        #var_34 = [0] * len(subcategorias)
        #var_45 = [0] * len(subcategorias)
        subset_geoamb = buscar_rango(df_geoamb, base_por_caracterizar.iloc[i]['x'],base_por_caracterizar.iloc[i]['y'])
        for j in range(0,len(subset_geoamb)):
            #j=0
            try:
                #xa = df_geoamb.iloc[j]['latitude']
                #ya = df_geoamb.iloc[j]['longitude']
                subcategoria = subset_geoamb.iloc[j]['subcategoria']
                distance = geopy.distance.vincenty((base_por_caracterizar.iloc[i]['x'], base_por_caracterizar.iloc[i]['y']),(subset_geoamb.iloc[j]['latitude'], subset_geoamb.iloc[j]['longitude'])).km
                if 1<distance:
                    pass
                elif 0<distance and distance<=0.5 :
                    lists_of_lists = [var_01, detectar_subcategoria(subcategoria,subcategorias)]
                    var_01 = [sum(x) for x in zip(*lists_of_lists)]
                elif 0.5<distance and distance<=1:
                    lists_of_lists = [var_12, detectar_subcategoria(subcategoria,subcategorias)]
                    var_12 = [sum(x) for x in zip(*lists_of_lists)]
                #elif 2<distance and distance<=3:
                #    lists_of_lists = [var_23, detectar_subcategoria(subcategoria,subset_geoamb)]
                #    var_23 = [sum(x) for x in zip(*lists_of_lists)]
                #elif 3<distance and distance<=4:
                #    lists_of_lists = [var_34, detectar_subcategoria(subcategoria,subset_geoamb)]
                #    var_34 = [sum(x) for x in zip(*lists_of_lists)]
                #elif 4<distance and distance<=5:
                #    lists_of_lists = [var_45, detectar_subcategoria(subcategoria,subset_geoamb)]
                #    var_45 = [sum(x) for x in zip(*lists_of_lists)]
            except:
                    pass
        var = [var_01, var_12]#, var_23,var_34, var_45]
        matriz[0] = id_data
        matriz[1] = base_por_caracterizar.iloc[i]['x']
        matriz[2] = base_por_caracterizar.iloc[i]['y']
        contador = 3
        for x in var:
            for val in x:
                matriz[contador] = val
                contador = contador + 1
        if len(matriz)<211:
            print('error')

        data = pd.DataFrame.from_dict(matriz).T
        file = 'D:/Modelos Tensorflow/Conjunto train y test/prose_PI.csv'
        data.to_csv(file, index=False, mode='a', header=False, encoding='latin1')

        p = p + 1
        stop = time.time()
        duration = stop-start
        val = 'Desarrollo:  ' + str(round((i-int(len(base_por_caracterizar)*val_inicial))/(int(len(base_por_caracterizar)*val_final)-int(len(base_por_caracterizar)*int(len(base_por_caracterizar)*val_inicial)))*100,1)) + '%' + '   tiempo:  ' + str(round(duration))
        print(val)
    return matriz


#df_etiqueta_traintest = pd.read_csv('D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas.csv', header=0, encoding = 'latin1')
df_etiqueta_traintest = pd.read_csv('D:/Modelos Tensorflow/SANTIAGO/id_prose_santiago(filtrado_autos).csv', header=0, encoding = 'latin1')
#contador_puntosinteres(df_etiqueta_traintest,subcategorias,df_geoamb,0,0.25)
#contador_puntosinteres(df_etiqueta_traintest,subcategorias,df_geoamb,0.25,0.50)
#contador_puntosinteres(df_etiqueta_traintest,subcategorias,df_geoamb,0.5,0.75)
contador_puntosinteres(df_etiqueta_traintest,subcategorias,df_geoamb,0.75,1) 
