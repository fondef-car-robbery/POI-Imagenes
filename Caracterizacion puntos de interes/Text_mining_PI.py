# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:30:48 2018

@author: Bgm9
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
from nltk.corpus import stopwords
from collections import Counter
import nltk


stop_words = stopwords.words('Spanish')
stop_words.extend(['ltda','limitada','spa','-','&','e.i.r.l.', 's.a.', '', 'sa', 'eirl', 'san', 'ltda.'
                   , 'sociedad','chile', '/', '2', '3', '5', '18a', '14a'])

def correccion_subcategoria(df):
    df['subcategoria'] = [str(x).replace(",","") for x in df['subcategoria']]
    df['subcategoria'] = [str(x).replace("-"," ") for x in df['subcategoria']]
    return df

df_civico = pd.read_csv('file:///C:/BGM/DireccionesCIVICO/puntosinteres.csv', header=0, encoding = 'utf-8', delimiter=';')

df_civico = correccion_subcategoria(df_civico)

subcategorias = sorted(list(set(df_civico['subcategoria'])))

sub_ranking = []
for subcategoria in subcategorias:
    #subcategoria = subcategorias[0] 
    texto_categoria = ''
    for i in range(0,len(df_civico)):
        if df_civico['subcategoria'][i] == subcategoria:    
            text = df_civico['nombre'][i]
            text = text.lower()
            texto_categoria = texto_categoria + ' ' + str(text)
    texto_categoria_split = texto_categoria.split(" ")
    filtered_sentence = [w for w in texto_categoria_split if not w in stop_words]
    counts = Counter(filtered_sentence)
    counts_top20 = counts.most_common(10)
    sub_ranking.append([subcategoria, counts_top20])


array_cate = []
for i in range(0,len(subcategorias)):   
    value = []
    subcategoria = sub_ranking[i][0]
    value.append(subcategoria)
    for j in range(0,len(sub_ranking[i][1])):
        palabra = sub_ranking[i][1][j][0]
        contador = sub_ranking[i][1][j][1]
        texto = palabra + ' (' + str(contador) + ')'
        value.append(texto)
    array_cate.append(value)
        
df = pd.DataFrame.from_records(array_cate)

file = 'C:/Users/Bgm9/Dropbox/CapHuida/caracterizacion/caracterizacion_subcategorias.csv'
df.to_csv(file, mode='w',index=False, header=False, encoding='latin1')

    
            



    