# -*- coding: utf-8 -*-
"""
Created on Sun May  6 02:44:48 2018

@author: Bgm9
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
import random
from random import uniform
import geopy.distance 
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

os.chdir("C:/BGM/Codigos python/Filtros DB prose/")
from Filtros import df_filtrar_por_tipovehiculo,eliminar_direcciones_sinnumeracion,eliminar_denuncias_erroneas, ID_Santiago_conimagenes

#####################################################################################################
'''                       HACEMOS UN CONTEO DE GRILLA                                             '''
#####################################################################################################

robo = pd.read_csv('file:///D:/Modelos Tensorflow/SANTIAGO/id_prose_santiago.csv', header=0, encoding = 'latin1')
filtro_uno = df_filtrar_por_tipovehiculo()
filtro_uno = filtro_uno.to_frame().reset_index()

filtro_dos = eliminar_direcciones_sinnumeracion()
filtro_tres = eliminar_denuncias_erroneas()
#filtro_cuatro = ID_Santiago_conimagenes() No es necesario

robo = pd.merge(robo, filtro_uno, left_on ='id_prose_santiago', right_on = 'id_prose' , how='inner')
robo = pd.merge(robo, filtro_dos, left_on ='id_prose_santiago', right_on = 'id_prose' , how='inner')
robo = pd.merge(robo, filtro_tres, left_on ='id_prose_santiago', right_on = 'id_prose' , how='inner')
#robo = pd.merge(robo, filtro_cuatro, left_on ='id_prose_santiago', right_on = 'id_prose' , how='inner')

#robo.to_csv('D:/Modelos Tensorflow/SANTIAGO/id_prose_santiago(filtrado_autos).csv', index=False, mode='w', header=True)

#df_geoprose = df_geoprose.rename(columns={'latitude': 'x', 'longitude': 'y'})
#robo = df_geoprose
#robo = robo.reset_index(drop=True)

# CON 70 hicimos la muestra final

min_x = min(robo['x'])
max_x = max(robo['x'])
divisionesx = 80#92 se usaba antes solo con el filtro_uno
pasox= (max_x-min_x)/divisionesx

min_y = min(robo['y'])
max_y = max(robo['y'])
divisionesy=math.ceil((max_y-min_y)/pasox)
pasoy = pasox

#divisionesy = 80#50
#pasoy= (max_y-min_y)/divisionesy

intervalosx = []
for i in range(0,divisionesx):
    min_intervalo =  min_x + pasox*i
    max_invervalo =  min_x + pasox*(i+1)
    inter = (min_intervalo, max_invervalo)
    intervalosx.append(inter)

    
intervalosy = []
for i in range(0,divisionesy):
    min_intervalo =  min_y + pasoy*i
    may_invervalo =  min_y + pasoy*(i+1)
    inter = (min_intervalo, may_invervalo)
    intervalosy.append(inter)

    
matriz = []
for i in range(divisionesy):
    matriz.append([0]*divisionesx) 
    
    
conteo = 0
for e in range(0,len(robo)):
    x1 = robo['x'][e]
    y1 = robo['y'][e]
    x = 0
    y = 0
    for interx in intervalosx:
        if x1 >= interx[0] and x1 <= interx[1]:
            for intery in intervalosy:
                if y1 >= intery[0] and y1 <= intery[1]:
                    matriz[y][x] = matriz[y][x] + 1
                    conteo = conteo + 1
                else:
                    y = y + 1
        else:
            x = x + 1
        
df = pd.DataFrame(matriz)
   
'''
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


dfList = df.values.tolist()
lista = []
for i in range(0,len(dfList)):
    lista = lista + dfList[i]
lista = remove_values_from_list(x, 0)


sns.distplot(lista, kde=False, rug=True)


#df = (df/len(robo))*100
'''

#####################################################################################################
'''                       RECUPERAMOS LOS SEGMENTOS DE MENOS-INDEFINIDO-MAS ROBO                                 '''
#####################################################################################################


paraaleatorio_norobo = []
## definimos los lugares de poco robo donde hay de uno a 5
corte_seguro = 5


cont_seguro = 0
cont_inseguro = 0
cont_neutro = 0

for i in range(0,divisionesx):
        for j in range(0,divisionesy):
            valor = df[i][j]
            if valor > 0 and valor <= corte_seguro:
                #print(i,j)
                #print(intervalosx[i])
                #print(intervalosy[j])
                paraaleatorio_norobo.append([intervalosx[i],intervalosy[j]])
                cont_seguro = cont_seguro + 1
                
paraaleatorio_robo = []
## definimos los lugares de poco robo donde hay de uno a 5
for i in range(0,divisionesx):
        for j in range(0,divisionesy):
            valor = df[i][j]
            if valor > corte_seguro:
                #print(i,j)
                #print(intervalosx[i])
                #print(intervalosy[j])
                paraaleatorio_robo.append([intervalosx[i],intervalosy[j]])  
                cont_inseguro = cont_inseguro + 1
                
string1 = 'El número de zonas seguras es: ' + str(cont_seguro)
string2 = 'El número de zonas inseguras es: ' + str(cont_inseguro)
print(string1)
print(string2)


p1 = (intervalosx[0][0], intervalosy[0][1])
p2 = (intervalosx[0][1] , intervalosy[0][1])
distance = round(geopy.distance.vincenty(p1, p2).km, 4)   
print(distance)

def punto_aleatorio(base):
   val = random.sample(base,1)[0]
   valx = val[0]
   valy = val[1]   
   return uniform(valx[0],valx[1]), uniform(valy[0],valy[1])

tamano_dataset = 10000
lista_norobo = []
for i in range(1,tamano_dataset):
    var= punto_aleatorio(paraaleatorio_norobo)
    lista_norobo.append(var)
    
base_norobo = pd.DataFrame(lista_norobo, columns=["x","y"])

lista_robo = []
for i in range(1,tamano_dataset):
    var= punto_aleatorio(paraaleatorio_robo)
    lista_robo.append(var)
    
base_robo = pd.DataFrame(lista_robo, columns=["x","y"])


plt.figure(figsize=(8, 8))
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='b') # Lugares no riesgosos
plt.show()      


# Esto no suma los 6400 porue hay muchos cuadrantes que no pertenecen a Santiago

#####################################################################################################
'''                       SUBCUADRANTES DE UN CUADRANTE                                           '''
#####################################################################################################

cuadrantes = paraaleatorio_norobo + paraaleatorio_robo

#i = 525 # i representa el cuadrante
#cuadrantes[i][0][0] # minimo x
#cuadrantes[i][0][1] # maximo x
#cuadrantes[i][1][0] # minimo y
#cuadrantes[i][1][1] # maximo y

subgrillas_all = []

for n in range(0,len(cuadrantes)):
    #n=0
    min_x = cuadrantes[n][0][0]
    max_x = cuadrantes[n][0][1] 
    divisionesx = 10#45
    #divisionesx = 4#45
    pasox= (max_x-min_x)/divisionesx
    
    min_y = cuadrantes[n][1][0]
    max_y = cuadrantes[n][1][1]
    divisionesy = 10
    #divisionesy = 4#50
    pasoy= (max_y-min_y)/divisionesy
    
    for i in range(0,divisionesx):
        min_intervalo =  min_x + pasox*i
        max_invervalo =  min_x + pasox*(i+1)
        interx = (min_intervalo, max_invervalo)
        for j in range(0,divisionesy):
            min_intervalo =  min_y + pasoy*j
            may_invervalo =  min_y + pasoy*(j+1)
            intery = (min_intervalo, may_invervalo)
            subgrillas_all.append([interx, intery])
            
len_grillas = len(subgrillas_all) 
submatriz = ([0]*len_grillas) 
    
robo = pd.read_csv('file:///D:/Modelos Tensorflow/SANTIAGO/id_prose_santiago.csv', header=0, encoding = 'latin1')
filtro_uno = df_filtrar_por_tipovehiculo()
filtro_uno = filtro_uno.to_frame().reset_index()

robo = pd.merge(robo, filtro_uno, left_on ='id_prose_santiago', right_on = 'id_prose' , how='inner')

p1 = (subgrillas_all[n][0][0], subgrillas_all[n][1][0])
p2 = (subgrillas_all[n][0][1] , subgrillas_all[n][1][0])
distance = round(geopy.distance.vincenty(p1, p2).km, 4)   
print(distance) 

for e in range(0,len(robo)):
    x1 = robo.iloc[e]['x']
    y1 = robo.iloc[e]['y']
    for n in range(0,len(subgrillas_all)):
        min_x = subgrillas_all[n][0][0]
        max_x = subgrillas_all[n][0][1] 
        min_y = subgrillas_all[n][1][0]
        max_y = subgrillas_all[n][1][1]
        if x1 >= min_x and x1 <= max_x and y1 >= min_y and y1 <= max_y:
            submatriz[n] = submatriz[n] + 1
            #robo = robo.drop(robo.index[e])
            #print(len(robo))
            
          
                
    #Ahora tendría que encontrar los intervalos donde no hay robos y etiquetarlo como robo y no robo
        
def punto_aleatorio(base):
   val = random.sample(base,1)[0]
   valx = val[0]
   valy = val[1]   
   return uniform(valx[0],valx[1]), uniform(valy[0],valy[1])

corte = 1
paraaleatorio_robo = []
paraaleatorio_norobo = []
for i in range(0,len_grillas):
        min_x = round(subgrillas_all[i][0][0],10)
        max_x = round(subgrillas_all[i][0][1],10)
        min_y = round(subgrillas_all[i][1][0],10)
        max_y = round(subgrillas_all[i][1][1],10)
        valor = submatriz[i]
        if valor < corte:
            paraaleatorio_norobo.append([(min_x,max_x),(min_y,max_y)])
        else:
            paraaleatorio_robo.append([(min_x,max_x),(min_y,max_y)])  

# la suma de esas dos listas tiene que ser 5880, si no esta malo

tamano_dataset = 10000
lista_norobo = []
for i in range(1,tamano_dataset):
    var= punto_aleatorio(paraaleatorio_norobo)
    lista_norobo.append(var)
    
base_norobo = pd.DataFrame(lista_norobo, columns=["x","y"])

lista_robo = []
for i in range(1,tamano_dataset):
    var= punto_aleatorio(paraaleatorio_robo)
    lista_robo.append(var)
    
base_robo = pd.DataFrame(lista_robo, columns=["x","y"])                


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.1, s =1.5, color='r') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='b') # Zona indefinida 
plt.show()    

#####################################################################################################
'''                                 ETIQUETAR ZONAS                                               '''
#####################################################################################################
#1. Cargar base random
#2. Verificar si punto de base random pertence al cuadrante de robo o no robo
    #1. Crear una funcion que verifique cuadrante por cuadrante
'''   
matriz_etiqueta = []
for i in range(divisionesy):
    matriz_etiqueta.append([0]*divisionesx) 
    
def definir_grilla_robo_norobo():
    for n_intery in range(0,len(intervalosx)):
        for n_interx in range(0,len(intervalosy)):
            if df[n_intery][n_interx] >= 5:
                matriz_etiqueta[n_intery][n_interx] = 1
    return matriz_etiqueta

matriz_etiqueta = definir_grilla_robo_norobo()
'''

df_etiqueta_traintest = pd.read_csv('D:/Modelos Tensorflow/Conjunto train y test/etapa1.csv', header=0, encoding = 'latin1')
df_etiqueta_traintest['label_zona'] = '' 
df_etiqueta_traintest['zona_cuadrante'] = '' 
df_etiqueta_traintest['label_f(zona)'] = '' 



for row_aleatorio in range(0,len(df_etiqueta_traintest)):
    #row_aleatorio = 6
    x1 = df_etiqueta_traintest['x'][row_aleatorio]
    y1 = df_etiqueta_traintest['y'][row_aleatorio]
    x = 0
    y = 0
    value = False
    for interx in intervalosx:
        if x1 >= interx[0] and x1 <= interx[1]:
            for intery in intervalosy:
                if y1 >= intery[0] and y1 <= intery[1] and matriz[y][x] > 5:
                    value = True
                    ubicacion = str(y) +','+ str(x)
                    #ubicacion = [y,x]
                    df_etiqueta_traintest.at[row_aleatorio, 'label_zona'] = 1
                    df_etiqueta_traintest.at[row_aleatorio, 'label_f(zona)'] = matriz[y][x]
                    df_etiqueta_traintest.at[row_aleatorio, 'zona_cuadrante'] = ubicacion
                    #print(ubicacion)
                    conteo = conteo + 1 
                if y1 >= intery[0] and y1 <= intery[1] and matriz[y][x] <= 5:
                    ubicacion = str(y) +','+ str(x)
                    #ubicacion = [y,x]
                    df_etiqueta_traintest.at[row_aleatorio, 'label_zona'] = 0
                    df_etiqueta_traintest.at[row_aleatorio, 'label_f(zona)'] = matriz[y][x]
                    df_etiqueta_traintest.at[row_aleatorio, 'zona_cuadrante'] = ubicacion
                    #print(ubicacion)
                else:
                    y = y + 1
        else:
            x = x + 1    
    #if value == False:
    #    df_etiqueta_traintest.loc[row_aleatorio,'label_zona'] = 0

    
# En la seccion anterior podria seleccionarse la cantidad de elementos que hay por cuadrante y 
    # evitar de esta forma que los puntos se encuentren demasiado cerca 
    
############################################
'''    GRAFICO DE ETIQUETAS ZONAS    '''
############################################    
## Deberia testear con un gráfico, para ver donde distribuyen los datos
base_robo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_zona'] == 1]
base_norobo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_zona'] == 0]
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.1, s =1.5, color='r', marker='.') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='b', marker='.') # Zona indefinida 
plt.show()    
    
#df_etiqueta_traintest.to_csv('D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_final_train_test_XY_etiqueta.csv', index=False, mode='w', header=True)

#####################################################################################################
'''                              ETIQUETAR LUGAR DE ROBO Y NO ROBO                                '''
#####################################################################################################


#df = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Random de Santiago.csv', header=0, encoding = 'latin1')
#row = 23
df_etiqueta_traintest['label_lugar'] = '' 
df_etiqueta_traintest['lugar_cuadrante'] = '' 
df_etiqueta_traintest['label_lugar>2'] = '' 
df_etiqueta_traintest['label_f(lugar)'] = '' 


# NO TENGO IDEA POR QUÉ HICE LO QUE ESTÁ COMENTADO
#divisionesx = 4#45
#pasox= 0.0004546235999995929
#divisionesx = math.ceil(((max_x-min_x)/pasox))

#min_y = min(min(intervalosy))
#max_y = max(max(intervalosy))
#divisionesy = 4#50
#pasoy= 0.0004546235999995929
#divisionesy = math.ceil((max_y-min_y)/pasoy)
#subgrillas_all = []

#for i in range(0,divisionesx):
#    min_intervalo =  min_x + pasox*i
#    max_invervalo =  min_x + pasox*(i+1)
#    interx = (min_intervalo, max_invervalo)
#    for j in range(0,divisionesy):
#        min_intervalo =  min_y + pasoy*j
#        may_invervalo =  min_y + pasoy*(j+1)
#        intery = (min_intervalo, may_invervalo)
#        subgrillas_all.append([interx, intery])


robo = robo.sort_values(by=['x'], ascending=True)
robo = robo.reset_index(drop=True)
submatriz = ([0]*len(subgrillas_all)) 
for e in range(0,len(robo)):
    print(e)
    x1 = robo.iloc[e]['x']
    y1 = robo.iloc[e]['y']
    for n in range(0,len(subgrillas_all)):
        min_x = subgrillas_all[n][0][0]
        max_x = subgrillas_all[n][0][1] 
        min_y = subgrillas_all[n][1][0]
        max_y = subgrillas_all[n][1][1]
        if x1 >= min_x and x1 <= max_x and y1 >= min_y and y1 <= max_y:
            submatriz[n] = submatriz[n] + 1
            break

import time
for row in range(0,len(df_etiqueta_traintest)):
    start = time.clock()
    px = df_etiqueta_traintest.iloc[row]['x']
    py = df_etiqueta_traintest.iloc[row]['y']
    for n in range(0,len(subgrillas_all)):
        min_x = subgrillas_all[n][0][0]
        max_x = subgrillas_all[n][0][1] 
        min_y = subgrillas_all[n][1][0]
        max_y = subgrillas_all[n][1][1]
        if (min_x<=px<=max_x) and (min_y<=py<=max_y):
                if submatriz[n] == 0:
                    df_etiqueta_traintest.at[row, 'label_lugar'] = 0 # norobo
                    df_etiqueta_traintest.at[row, 'label_lugar>2'] = 0 # norobo
                    df_etiqueta_traintest.at[row, 'label_f(lugar)'] = submatriz[n]                  
                    df_etiqueta_traintest.at[row, 'lugar_cuadrante'] = n # norobo
                    # 0 es bajoriesgo
                    value = True
                    break
                elif submatriz[n] > 0:
                    if submatriz[n] >= 2:
                        df_etiqueta_traintest.at[row, 'label_lugar'] = 1 # robo
                        df_etiqueta_traintest.at[row, 'label_lugar>2'] = 2 # norobo
                        df_etiqueta_traintest.at[row, 'label_f(lugar)'] = submatriz[n]      
                        df_etiqueta_traintest.at[row, 'lugar_cuadrante'] = n
                    else:
                        df_etiqueta_traintest.at[row, 'label_lugar'] = 1 # robo
                        df_etiqueta_traintest.at[row, 'label_lugar>2'] = 1 # norobo
                        df_etiqueta_traintest.at[row, 'label_f(lugar)'] = submatriz[n]      
                        df_etiqueta_traintest.at[row, 'lugar_cuadrante'] = n
                    # 0 es altoriesgo
                    value = True
                    break
    end = time.clock()
    print(end-start)

df_etiqueta_traintest.to_csv('D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', index=False, mode='w', header=True)
## la v2 contiene nuevas etiquetas (label_lugar>2, label_f(lugar), label_f(zona))
    

############################################
'''    GRAFICO DE ETIQUETAS LUGARES      '''
############################################  
#df_etiqueta_traintest = pd.read_csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_final_train_test_XY_etiqueta_lugar.csv', header=0, encoding = 'latin1')

### Lugares con mas de un robo
base_robo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_lugar'] == 1]
base_norobo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_lugar'] == 0]
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.1, s =1.5, color='r', marker=',') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='b', marker='.') # Zona indefinida 
plt.show()   

### Lugares con mas de dos robos
base_robo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_lugar>2'] == 1]
base_norobo = df_etiqueta_traintest.loc[df_etiqueta_traintest['label_lugar>2'] == 0]
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.1, s =1.5, color='r', marker=',') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='b', marker='.') # Zona indefinida 
plt.show()   

### Segmentando por 0 robos, 1 robo y 2 robos o mas
base = df_etiqueta_traintest[df_etiqueta_traintest['label_f(lugar)'] != '']
base['label_f(lugar)']  = base['label_f(lugar)'].astype('int')

base_robo2omas = base.loc[base['label_f(lugar)'] > 1]
base_robo = base.loc[base['label_f(lugar)'] == 1]
base_norobo = base.loc[base['label_f(lugar)'] == 0]
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(base_robo2omas['x'],base_robo2omas['y'], alpha=0.5, s =1.5, color='r', marker=',') # Zona indefinida 
plt.scatter(base_robo['x'],base_robo['y'], alpha=0.2, s =1.5, color='k', marker=',') # Lugares riesgosos
plt.scatter(base_norobo['x'],base_norobo['y'], alpha=0.1, s =1.5, color='lightblue', marker='.') # Zona indefinida 
plt.show()  



          

























