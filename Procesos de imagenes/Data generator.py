# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:56:04 2018

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

#directorio_save = 'D:/Modelos Tensorflow/Conjunto train y test/data_augmentation/'
directorio_save = 'C:/data_augmentation/'
archivos = os.listdir(directorio_save)
ids = []
for archivo in archivos:
    #archivo = archivos[0]
    p = archivo.find('_')
    id_rec = archivo[0:p]
    ids.append(int(id_rec))
ids = set(ids)




#img = Image.open('D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/143751/180/gsv_0.jpg')
#df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')

directorio_fotos = 'D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes'
directorio_save = 'C:/data_augmentation/'
df_ids = pd.read_csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas_v2.csv', header=0, encoding = 'latin1')

df_ids['sin_hora_siniestro'] = df_ids['id']
id_label_lugar = df_ids.loc[df_ids['label_lugar'] == 1]

#sess.run(tf.global_variables_initializer())

#tf.reset_default_graph()

for i in range(0,len(id_label_lugar)):
    id_foto = id_label_lugar.iloc[i]['id']
    if int(id_foto) not in ids:
        lista_imagenes = []
        angulo_imagenes = []
        try:
            f = str(int(id_foto)) + '/0/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            #img0 = Image.open(directorio_archivo)
            img0 = cv2.imread(directorio_archivo) #load rgb image
            if type(img0) is not type(None):
                lista_imagenes.append(img0)
                angulo_imagenes.append('0')
        except:
            pass
        try:
            f = str(int(id_foto)) + '/90/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            #img90 = Image.open(directorio_archivo)
            img90 = cv2.imread(directorio_archivo)
            if type(img90) is not type(None):
                lista_imagenes.append(img90)
                angulo_imagenes.append('90')
        except:
            pass
        try:
            f = str(int(id_foto)) + '/180/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            #img180 = Image.open(directorio_archivo)
            img180 = cv2.imread(directorio_archivo)
            if type(img180) is not type(None):
                lista_imagenes.append(img180)
                angulo_imagenes.append('180')
        except:
            pass
        try:
            f = str(int(id_foto)) + '/270/gsv_0.jpg'
            directorio_archivo = os.path.join(directorio_fotos, f)
            #img270 = Image.open(directorio_archivo)
            img270 = cv2.imread(directorio_archivo)
            if type(img270) is not type(None):
                lista_imagenes.append(img270)
                angulo_imagenes.append('270')
        except:
            pass
        start = time.time()
        #with tf.Session() as sess:
        for c in range(0,len(lista_imagenes)):
                
                img = lista_imagenes[c]
                angulo = angulo_imagenes[c]
                img = cv2.resize(img, (224, 224)) 

                #sess.run(tf.global_variables_initializer())

                #img = np.asarray(img, dtype=np.float32) # SOLO PARA INSTANCIA
                #x = tf.image.resize_images(img, [224,224])
                #max_delta = 0.30
                #img_tf_brightness = tf.image.random_brightness(img, max_delta, seed=None) # Brillo funcionando random
                #lower = 0.25
                #upper = 0.50
                #img_tf_contrast = tf.image.random_contrast(img, lower, upper, seed=None) # Contraste random    
                #lower = 0.25
                #upper = 0.50
                #img_tf_saturation = tf.image.random_saturation(img, lower, upper, seed=None) # Saturacion random
                #max_delta = 0.30
                #img_tf_random_hue = tf.image.random_hue(img, max_delta, seed=None) # Hue random
                #img_tf_flip = tf.image.flip_left_right(img) # Voltear random
                
                #img_tf_flip = tf.Variable(x)
                #img = cv2.imread('D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/143725/180/gsv_0.jpg') #load rgb image

                
                # Imagen desenfocada
                img0 = cv2.GaussianBlur(img,(7,7),0)
                #cv2.imshow('i',img0)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                #imagen en blanco y negro
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img1 = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                #cv2.imshow('i',grayrgb)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                
                #median = cv2.medianBlur(img,9)
                #cv2.imshow('i',median)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                #fliping imagen
                img2 = cv2.flip(img, flipCode=1)
                #cv2.imshow('i',gamma_corrected)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                # Imagen oscura
                img3 = exposure.adjust_gamma(img, 2)
                #cv2.imshow('i',img3)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                # Imagen Clara
                img4 = exposure.adjust_gamma(img, 0.4)
                #cv2.imshow('i',img4)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                alpha = float(1.7)     # Simple contrast control '* Enter the alpha value [1.0-3.0]: '
                beta = int(20)             # Simple brightness control 'Enter the beta value [0-100]: '
                mul_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
                img6 = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta
                   
                #cv2.imshow('original_image', img)
                #cv2.imshow('new_image',img6)
                
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                ## Efecto similar a HUE
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                h += 50 # 4
                s += 0 # 5
                v += 0 # 6
                final_hsv = cv2.merge((h, s, v))
                img5 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                #cv2.imshow('i',img2)
                #cv2.waitKey(0)
                #cv2.destroyWindow('i')
                
                #cv2.imshow('original', img)
                #cv2.imshow('1',img0)
                #cv2.imshow('2', img1)
                #cv2.imshow('3',img2)
                #cv2.imshow('4', img3)
                #cv2.imshow('5',img4)
                #cv2.imshow('6', img5)
                #cv2.imshow('7',img6)
                
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                
                #sess.close()
                #init = tf.initialize_all_variables()
                #sess = tf.Session()
                #sess.run(init)
                #img1 = sess.run(img_tf_brightness)
                #img2 = sess.run(img_tf_contrast)
                #img3 = sess.run(img_tf_saturation)
                #img4 = sess.run(img_tf_random_hue)
                #img5 = sess.run(img_tf_flip)
                
                guardar_im0 = directorio_save + str(id_foto) + '_' + angulo + 'g' + '.jpg'
                guardar_im1 = directorio_save + str(id_foto) + '_' + angulo + 'b' + '.jpg'
                guardar_im2 = directorio_save + str(id_foto) + '_' + angulo + 'c' + '.jpg'
                guardar_im3 = directorio_save + str(id_foto) + '_' + angulo + 's' + '.jpg'
                guardar_im4 = directorio_save + str(id_foto) + '_' + angulo + 'h' + '.jpg'
                guardar_im5 = directorio_save + str(id_foto) + '_' + angulo + 'f' + '.jpg'
                guardar_im6 = directorio_save + str(id_foto) + '_' + angulo + 'bc' + '.jpg'
                 
                cv2.imwrite(guardar_im0, img0)
                cv2.imwrite(guardar_im1, img1)
                cv2.imwrite(guardar_im2, img2)
                cv2.imwrite(guardar_im3, img3)
                cv2.imwrite(guardar_im4, img4)
                cv2.imwrite(guardar_im5, img5)
                cv2.imwrite(guardar_im6, img6)
        end = time.time()
        print(end - start)
    else:
        print('caso')

                    