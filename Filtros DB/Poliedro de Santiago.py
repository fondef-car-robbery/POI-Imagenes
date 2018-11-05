# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:57:44 2018

@author: Bgm9
"""
#####################################################################################################
'''                       POLIEDRO DE LA CIUDAD DE SANTIAGO                               '''
#####################################################################################################    
import pandas as pd
import re

pd.options.display.float_format = '{:.9f}'.format
df = pd.read_csv('C:/BGM/Santiago/Poliedro Comunas de Santiago.csv', header=0, encoding = 'latin1')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union

#convertimos el dataset en un vector (x,y) para consturir el poliedro
poligono =[]
comuna = []
for i in range(0,len(df)):   
             a = df['longitude'][i]
             b = df['latitude'][i]
             c = df['comuna'][i]
             poligono.append((a,b))
             comuna.append(c)
        
list(set(comuna))

Pudahuel =[]
Macul =[]
Nunoa =[]
LasCondes =[]
SanJoaquin =[]
LaGranja =[]
LaCisterna =[]
QuintaNormal =[]
Recoleta =[]
PadreHurtado =[]
LoEspejo =[]
Conchali =[]
LoBarnechea =[]
LaFlorida =[]
Maipu =[]
SanBernardo =[]
Santiago =[]
SanMiguel =[]
SanRamon =[]
CerroNavia =[]
Cerrillos =[]
Quilicura =[]
Independencia =[]
PedroAguirreCerda =[]
PuenteAlto =[]
Huechuraba =[]
LaPintana =[]
LoPrado =[]
Penalolen =[]
Vitacura =[]
EstacionCentral =[]
Providencia =[]
Renca =[]
ElBosque =[]

for i in range(0,len(df)):
     b = df['longitude'][i]
     a = df['latitude'][i]
     if df['comuna'][i] == 'Pudahuel':
         Pudahuel.append((a,b))
     elif df['comuna'][i] == 'Macul':
         Macul.append((a,b))
     elif df['comuna'][i] == 'Las Condes':
         LasCondes.append((a,b))
     elif df['comuna'][i] == 'San Joaquin':
         SanJoaquin.append((a,b))
     elif df['comuna'][i] == 'La Granja':
         LaGranja.append((a,b))
     elif df['comuna'][i] == 'La Cisterna':
         LaCisterna.append((a,b))
     elif df['comuna'][i] == 'Quinta Normal':
         QuintaNormal.append((a,b))
     elif df['comuna'][i] == 'Nunoa':
         Nunoa.append((a,b))
     elif df['comuna'][i] == 'Recoleta':
         Recoleta.append((a,b))
     elif df['comuna'][i] == 'Padre Hurtado':
         PadreHurtado.append((a,b))
     elif df['comuna'][i] == 'Lo Espejo':
         LoEspejo.append((a,b))
     elif df['comuna'][i] == 'Conchali':
         Conchali.append((a,b))
     elif df['comuna'][i] == 'Lo Barnechea':
         LoBarnechea.append((a,b))
     elif df['comuna'][i] == 'La Florida':
         LaFlorida.append((a,b))
     elif df['comuna'][i] == 'Maipu':
         Maipu.append((a,b))
     elif df['comuna'][i] == 'San Bernardo':
         SanBernardo.append((a,b))
     elif df['comuna'][i] == 'Santiago':
         Santiago.append((a,b))
     elif df['comuna'][i] == 'San Miguel':
         SanMiguel.append((a,b))
     elif df['comuna'][i] == 'San Ramon':
         SanRamon.append((a,b))
     elif df['comuna'][i] == 'Cerro Navia':
         CerroNavia.append((a,b))
     elif df['comuna'][i] == 'Cerrillos':
         Cerrillos.append((a,b))
     elif df['comuna'][i] == 'Quilicura':
         Quilicura.append((a,b))
     elif df['comuna'][i] == 'Independencia':
         Independencia.append((a,b))
     elif df['comuna'][i] == 'Pedro Aguirre Cerda':
         PedroAguirreCerda.append((a,b))
     elif df['comuna'][i] == 'Penalolen':
         Penalolen.append((a,b))
     elif df['comuna'][i] == 'Puente Alto':
         PuenteAlto.append((a,b))
     elif df['comuna'][i] == 'Huechuraba':
         Huechuraba.append((a,b))
     elif df['comuna'][i] == 'La Pintana':
         LaPintana.append((a,b))
     elif df['comuna'][i] == 'Lo Prado':
         LoPrado.append((a,b))
     elif df['comuna'][i] == 'Vitacura':
         Vitacura.append((a,b))
     elif df['comuna'][i] == 'Estacion Central':
         EstacionCentral.append((a,b))
     elif df['comuna'][i] == 'Providencia':
         Providencia.append((a,b))
     elif df['comuna'][i] == 'Renca':
         Renca.append((a,b))
     elif df['comuna'][i] == 'El Bosque':
         ElBosque.append((a,b))


Pudahuel =Polygon(Pudahuel)
Macul =Polygon(Macul)
Nunoa =Polygon(Nunoa)
LasCondes =Polygon(LasCondes)
SanJoaquin =Polygon(SanJoaquin)
LaGranja =Polygon(LaGranja)
LaCisterna =Polygon(LaCisterna)
QuintaNormal =Polygon(QuintaNormal)
Recoleta =Polygon(Recoleta)
PadreHurtado =Polygon(PadreHurtado)
LoEspejo =Polygon(LoEspejo)
Conchali =Polygon(Conchali)
LoBarnechea =Polygon(LoBarnechea)
LaFlorida =Polygon(LaFlorida)
Maipu =Polygon(Maipu)
SanBernardo =Polygon(SanBernardo)
Santiago =Polygon(Santiago)
SanMiguel =Polygon(SanMiguel)
SanRamon =Polygon(SanRamon)
CerroNavia =Polygon(CerroNavia)
Cerrillos =Polygon(CerroNavia)
Quilicura =Polygon(Quilicura)
Independencia =Polygon(Independencia)
PedroAguirreCerda =Polygon(PedroAguirreCerda)
PuenteAlto =Polygon(PuenteAlto)
Huechuraba =Polygon(Huechuraba)
LaPintana =Polygon(LaPintana)
LoPrado =Polygon(LoPrado)
Penalolen =Polygon(Penalolen)
Vitacura =Polygon(Vitacura)
EstacionCentral =Polygon(EstacionCentral)
Providencia =Polygon(Providencia)
Renca =Polygon(Renca)
ElBosque =Polygon(ElBosque)


GranSantiago = []

lista_comunas_poly = [Pudahuel,
 Macul,
 LasCondes,
 SanJoaquin,
 LaGranja,
 LaCisterna,
 QuintaNormal,
 Nunoa,
 Recoleta,
 PadreHurtado,
 LoEspejo,
 Conchali,
 LoBarnechea,
 LaFlorida,
 Maipu,
 SanBernardo,
 Santiago,
 SanMiguel,
 SanRamon,
 CerroNavia,
 Cerrillos,
 Quilicura,
 Independencia,
 PedroAguirreCerda,
 Penalolen,
 PuenteAlto,
 Huechuraba,
 LaPintana,
 LoPrado,
 Vitacura,
 EstacionCentral,
 Providencia,
 Renca,
 ElBosque]

GranSantiago = cascaded_union(lista_comunas_poly)

Polygon(poligono)




geolocali[848]
#print(polygon.bounds)
pertenecencia = []
for i in range(0,len(geolocali)):
    point = Point(geolocali[i])
    pertenecencia.append(polygon.contains(point))