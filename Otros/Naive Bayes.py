# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:35:56 2018

@author: bruno
"""
import sys

import nltk
import random
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv

def juntar_lista(lis):
    concatenar = []    
    for i in lis:
        for j in lis:
            concatenar = concatenar + j
    return concatenar

palabras = []
var = []
caso = []
reader = csv.reader( open('C:/BGM/clasificación de texto/dataset_textclasification_python.csv', 'r'), delimiter=';', dialect='excel')
for row in reader: 
    palabras.append(row[0].split())
    var = [row[0],'']
    caso.append(var)
    
data_clasificada = caso[0:400]    

"""
import pandas as pd
df = pd.DataFrame(data_clasificada, columns=['descripcion','etiqueta'])
df.to_csv('/home/bruno/Documentos/Python - Memoria/Naive Bayes/data_etiquetada.csv', index=False, mode='a')

# randomizamos shuffle
from random import shuffle
random.shuffle(caso)

caso_dataframe = pd.DataFrame(caso, columns=['descripcion','etiqueta'])
all_words = juntar_lista(list(caso_dataframe['descripcion'][0:10]))
"""
# Definimos las categorias
categorias = ['hurto','otra']
# Construyo un string gigante con todos los relatos
concat = ""
for i in data_clasificada:
    concat = concat + str(i[0])
# Separo cada palabra del string gigante en una lista gigante
palabra_documento = concat.split()
stop_words = set(stopwords.words('spanish'))
# Eliminamos los stopwords
palabra_documento_filtrado = []
for w in palabra_documento:
    if w.lower() not in stop_words:
        palabra_documento_filtrado.append(w) 
# Sacamos las frecuencias de aparición de cada palabra
all_words = nltk.FreqDist(palabra_documento_filtrado) ## Obtenemos las frecuencias de aparicicón de cada palabra del corpus
word_features = list(all_words.keys())[0:1000] # Obtiene un subconjunto de las 100 palabras clave (mayor frecuencia) 
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def eliminar_stopwords(valor):
    resultado = [] 
    for w in valor:
        if w.lower() not in stop_words:
            from nltk.stem import SnowballStemmer
            #w = stemmer.stem(w)
            resultado.append(w)
    return resultado

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
## renombrar los otros 

data_clasificadav2 = []
for i in data_clasificada:
    if i[1] != 'hurto':
        data_clasificadav2.append([i[0],'otro'])
    else:
        data_clasificadav2.append([i[0],i[1]])

## Creamos el documento
document = []
for i in data_clasificadav2:
    lista_palabras = i[0].split()
    lista_palabras = eliminar_stopwords(lista_palabras)
    clasificacion = i[1]
    document.append([lista_palabras,clasificacion])

featuresets = [(find_features(relato), category) for (relato, category) in document]   

train_set, test_set = featuresets[0:300], featuresets[301:400]
classifier = nltk.NaiveBayesClassifier.train(train_set)
## Pendiente probar con SVM.
""" No CARGA dice que noes está instalado correctamente
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
clas2 = nltk.classify.svm.SvmClassifier(train_set)
"""
print(nltk.classify.accuracy(classifier, test_set)) ## Accuracy del 78%

import hunspell
    
## dataset_corpus tiene que se un vector con todos los comentarios
# cada w del corpus debe corresponder a una palabra de cada elemento del vector
for relato in vector_relatos:
    for palabra in relato.split():
        all_words.append(palabra.lower())
        

# Acá debería crear una lista de lista que contenga cada relato y dentro un vector de las
    # palabras contendias en el relato y a esto se le aplica la función 
dataset_a_evaluar = []
categorias = ['robo','hurto']

def find_keyfeatures(alldata):
    word = set(alldata)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    
featuresets = [(find_features(rev), categorias) for (rev, categorias) in alldata]
    

# set that we'll train our classifier with
training_set = featuresets[:1900]
# set that we'll test against.
testing_set = featuresets[1900:]


    

word_features = list(all_words.keys()

classifier = nltk.NaiveBayesClassifier.train(training_set)



print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


category = ['Hurto','Robo']

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = movie_reviews.words()
all_words[1000]
classifier.show_most_informative_features(15)
