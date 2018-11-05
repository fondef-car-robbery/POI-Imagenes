# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:59:18 2018

@author: bruno
"""


"""
import pip
name = 'googlemaps'
pip.main(['install', name])
"""
import sys
import geopy
import csv
import googlemaps

gmaps = googlemaps.Client(key='AIzaSyBVFaz_381Vnj0EpdrioXmcr7WuWSAPDkA') ## bgmgomez (fundiendo) 11
gmaps = googlemaps.Client(key='AIzaSyAjUqDVWpXRSdcMa1trgWui7MkRM4OutrU') ## brunoinstituto
gmaps = googlemaps.Client(key='AIzaSyBJpTYsdNJiuq0IjPe1qRUqTRqO0xMjU-4') ## Arribaelnem
gmaps = googlemaps.Client(key='AIzaSyBVVOojSozspJYsng9UeswJHwhOm4NJvsQ') ## memoriauchile2018
gmaps = googlemaps.Client(key='AIzaSyDXw3aHTrh-VG9y9whQXliL4vicryuEIo0') ## brugomez2017
gmaps = googlemaps.Client(key='AIzaSyA8fykjqUbyJg6gELH1xt_biiKvssfBK3o') ## Serviciosintarjeta
gmaps = googlemaps.Client(key='AIzaSyAi3wquv6U5mdWYs2zJvcK_XEbEk7Fpek8') ## anamariamom2019
gmaps = googlemaps.Client(key='AIzaSyBTJcNP6Lbk-EsOsD4jDxyredVDJL00fp4') ## brugomez2018 8

# Función para encontrar el X,Y del robo usando la API de Google 
def buscadorxy(direcciones_v2):
    rec_direc =[]
    r = 0
    num = len(direcciones_v2)
    for i in direcciones_v2:
        if i != "":
            try:
                print(i)
                geocode_result = gmaps.geocode(str(i))
                try:        
                    latitud = geocode_result[0]['geometry']['location']['lat']
                    longitud = geocode_result[0]['geometry']['location']['lng']
                except:
                    latitud = ""
                    longitud = ""
                r = r + 1
                rec_direc.append([i,latitud,longitud])
                print([latitud, longitud])
                value =  round(r/num*100,2)
                print(str(value)+'%')
            except:
                latitud = ""
                longitud = ""
                r = r + 1
                value =  round(r/num*100,2)
                rec_direc.append([i,latitud,longitud])
                print('Error geocode')
                print(str(value)+'%')
    return rec_direc




## Agrego las direcciones
direcciones = []
reader = csv.reader( open('C:\BGM\DireccionesCIVICO\puntosinteres.csv', 'r'), delimiter=';', dialect='excel')
for row in reader: 
    var = row[1] + ', Santiago' + ', Region Metropolitana' + ', Republica de Chile'
    direcciones.append(var)

xy_prediccion = []

direcciones_1 = direcciones[0:2000] ## ejecutado
direcciones_2 = direcciones[2001:4450] ## ejecutado
direcciones_3 = direcciones[4451:6900] ## ejecutado
direcciones_4 = direcciones[6901:9300] ## ejecutado
direcciones_5 = direcciones[9301:11700] ## ejecutado
direcciones_6 = direcciones[11701:14101] ## ejecutado
direcciones_7 = direcciones[14101:16500] ## ejecutado
direcciones_8 = direcciones[16501:18900] ## ejecutado
direcciones_9 = direcciones[18901:21300] ## ejecutado
direcciones_10 = direcciones[21301:23700] ## ejecutado
direcciones_11 = direcciones[23701:26100] ## ejecutado
direcciones_12 = direcciones[26101:28500] ## ejecutado
direcciones_13 = direcciones[28501:30900] ## ejecutado
direcciones_14 = direcciones[30901:33300] ## ejecutado
direcciones_15 = direcciones[33301:35700] ## ejecutado
direcciones_16 = direcciones[35701:38100] ## ejecutado
direcciones_17 = direcciones[38101:40500] ## ejecutado
direcciones_18 = direcciones[40501:42900] ## ejecutado
direcciones_19 = direcciones[42901:45300] ## ejecutado
direcciones_20 = direcciones[45301:47700] ## ejecutado
direcciones_21 = direcciones[47701:50100] ## ejecutado
direcciones_22 = direcciones[50101:52500] ## ejecutado
direcciones_23 = direcciones[52501:54900] ## ejecutado
direcciones_24 = direcciones[54901:57300] ## ejecutado
direcciones_25 = direcciones[57301:59700] ## ejecutado
direcciones_26 = direcciones[59701:62100] ## ejecutado
direcciones_27 = direcciones[62101:64500] ## ejecutado
direcciones_28 = direcciones[64501:66900] ## ejecutado
direcciones_29 = direcciones[66901:69300] ## ejecutado
direcciones_30 = direcciones[69301:71700] ## ejecutado
direcciones_31 = direcciones[71701:74100] ## ejecutado
direcciones_32 = direcciones[74101:76500] ## ejecutado
direcciones_33 = direcciones[76501:78900] ## ejecutado
direcciones_34 = direcciones[78901:81300] ## ejecutado
direcciones_35 = direcciones[81301:83700] ## ejecutado
direcciones_36 = direcciones[83701:86100] ## ejecutado
direcciones_37 = direcciones[86101:88500] ## ejecutado
direcciones_38 = direcciones[88501:90900] ## ejecutado
direcciones_39 = direcciones[90901:93300] ## ejecutado
direcciones_40 = direcciones[93300:95700] ## ejecutado
direcciones_41 = direcciones[95700:98100] ## ejecutado
direcciones_42 = direcciones[98100:100500] ## ejecutado
direcciones_43 = direcciones[100500:102900] ## ejecutado
direcciones_44 = direcciones[102900:105300] ## ejecutado
direcciones_45 = direcciones[105300:107700] ## ejecutado
direcciones_46 = direcciones[107700:110100] ## ejecutado
direcciones_47 = direcciones[110100:112500] ## ejecutado
direcciones_48 = direcciones[112500:114900] ## ejecutado
direcciones_49 = direcciones[114900:117300] ## ejecutado
direcciones_50 = direcciones[117300:119700] ## ejecutado
direcciones_51 = direcciones[119700:122100] ## ejecutado
direcciones_52 = direcciones[122100:124500] ## ejecutado
direcciones_53 = direcciones[124500:126900] ## ejecutado
direcciones_54 = direcciones[126900:129300] ## ejecutado
direcciones_55 = direcciones[129300:131700] ## ejecutado
direcciones_56 = direcciones[131700:134100] ## ejecutado
'''
n = 150900
for i in range(1,10):
    print(n)
    n = n + 2400
'''

xy_prediccion = []
geolocator = Nominatim()
contador = 0
for i in direcciones_18:
    try:
        location = geolocator.geocode(i)
        xy_prediccion.append([i, location.latitude, location.longitude])
        print(contador)
        contador = contador + 1
    except:
        print('Error')
        xy_prediccion.append([i,'', ''])
        print(contador)
        contador = contador + 1
     

xy_prediccion54 = buscadorxy(direcciones_54) 
xy_prediccion55 = buscadorxy(direcciones_55)     
xy_prediccion56 = buscadorxy(direcciones_56)  
        
import pandas as pd
df = pd.DataFrame(xy_prediccion54, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion55, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion56, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion50, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example117300-119700.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion51, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example119700-122100.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion52, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example122100-124500.csv', index=False, mode='a')

import pandas as pd
df = pd.DataFrame(xy_prediccion53, columns=['direccion','latitude','longitude'])
df.to_csv('C:/BGM/DireccionesCIVICO/example124500-126900.csv', index=False, mode='a')


    
    