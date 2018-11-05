# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 03:19:53 2018

@author: Bgm9
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import os
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import scipy.misc
from skimage import data, io, filters
from PIL import Image
from random import shuffle
from resizeimage import resizeimage
from numpy import array
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance
import time
import scipy.misc
import imageio
from skimage import data, exposure, img_as_float
from skimage import restoration
from skimage.morphology import disk
import cv2


#directorio_fotos = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/'
directorio_fotos = 'C:/Users/Administrator/Documents/Imagenes'# # Directorio de amazon

lista_folders = os.listdir(directorio_fotos)
for num in range(0,len(lista_folders)):
        #num = 1
        id_foto = lista_folders[num]
        try:
            f = str(int(id_foto)) + '/gsv_pano.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            img = cv2.resize(img, (224*4, 224)) 
            cv2.imwrite(directorio_archivo, img)
        except:
            pass
        try:
            f = str(int(id_foto)) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            img = cv2.resize(img, (224, 224)) 
            cv2.imwrite(directorio_archivo, img)
        except:
            pass
        try:
            f = str(int(id_foto)) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo) 
            img = cv2.resize(img, (224, 224)) 
            cv2.imwrite(directorio_archivo, img)
        except:
            pass
        try:
            f = str(int(id_foto)) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            img = cv2.resize(img, (224, 224)) 
            cv2.imwrite(directorio_archivo, img)
        except:
            pass
        try:
            f = str(int(id_foto)) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            img = cv2.imread(directorio_archivo)
            img = cv2.resize(img, (224, 224)) 
            cv2.imwrite(directorio_archivo, img)
        except:
            pass
        
        
        
        
        
        
        
        
        
        
        
        
        