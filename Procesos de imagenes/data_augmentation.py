# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:16:16 2018

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

imageSource = 'images/messi5.jpg'
img = cv2.imread(imageSource)
# Use the following commands to perform random crops
#original_size = [224, 224]
#crop_size = [224, 224, 3]
#seed = np.random.randint(1234)
#x = tf.random_crop(img, size = crop_size, seed = seed)
#output = tf.image.resize_images(x, size = original_size)
#img_tf = tf.Variable(output)

img = Image.open('D:/Modelos Tensorflow/base de datos - Random de Santiago/Imagenes/14118/180/gsv_0.jpg')

#import random
img = np.asarray(img, dtype=np.float32)
image_flip = cv2.flip(img, 0)
cv2.imshow('Original', image_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()

x = tf.image.flip_left_right(img) #Funcionando Fliping
img_tf = tf.cast(x, dtype=tf.float32)
X_flip = np.array(img_tf, dtype = np.float32)


r = resizeimage.resize_cover(img, [tamagno_img, tamagno_img])
img = array(r)
img = np.asarray(img, dtype=np.float32)
imag_test.append(img)
idimagen_test.append(int(test.iloc[row]['id']))
valor_label = test.iloc[row][nombre_etiqueta]
valor_label = asignar_label(valor_label, nombre_etiqueta)
etiquetas_test.append(valor_label)




x = tf.image.resize_images(
    img,
    [224,224],
)
img_resize = tf.Variable(x)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im_r = sess.run(img_resize).astype('uint8')

max_delta = 0.50
x = tf.image.random_brightness(im_r, max_delta, seed=None) # Brillo funcionando random
img_tf_brightness = tf.Variable(x)

lower = 0.25
upper = 0.50
x = tf.image.random_contrast(im_r, lower, upper, seed=None) # Contraste random
img_tf_contrast = tf.Variable(x)

lower = 0.25
upper = 0.50
x = tf.image.random_saturation(im_r, lower, upper, seed=None) # Saturacion random
img_tf_saturation = tf.Variable(x)

max_delta = 0.30
x = tf.image.random_hue(im_r, max_delta, seed=None) # Hue random
img_tf_random_hue = tf.Variable(x)

x = tf.image.flip_left_right(im_r) # Voltear random
img_tf_flip = tf.Variable(x)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im1 = sess.run(img_tf_brightness)#.astype('float32')
im2 = sess.run(img_tf_contrast)#.astype('float32')
im3 = sess.run(img_tf_saturation)#.astype('float32')
im4 = sess.run(img_tf_random_hue)#.astype('float32')
im5 = sess.run(img_tf_flip)#.astype('float32')
plt.imshow(im_r)
plt.imshow(im1)
plt.imshow(im2)
plt.imshow(im3)
plt.imshow(im4)
plt.imshow(im5)


r = resizeimage.resize_cover(img, [224, 224])
img = array(r)
img = np.asarray(img, dtype=np.float32)

#from skimage import restoration
#img2 = restoration.denoise_tv_chambolle(img, weight=0.1) # lo deja como ne blanco y negro
#img3 = filters.gaussian(img, sigma=2) # Como que lo pixelea
#plt.imshow(img)

#from skimage import exposure
#img4 = exposure.equalize_hist(img)

#crop_size = [224, 224, 3]
#seed = np.random.randint(1234)
#x = tf.random_crop(img, size = crop_size, seed = seed)
#output = tf.image.ResizeMethod(x, size = crop_size)

#