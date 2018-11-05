Codigos de la carpeta
1. concatenar imagenes - random santiago:
	[Util]
	Este codigo lee las imágenes separadas de 0, 90, 180 y 270 grados de la carpeta de IMÁGENES (label_directory)
	y construye una imágen panorámica y luego la guardar en un directorio (guardar_en)

2. data_augmentation:
	[No utilizado]
	Este código contiene algunos efectos que se pueden aplicar a las imágenes usando la librería de 
	tensorflow, sin embargo, no se recomienda utilizar porque el tiempo de procesamiento para generar las 
	imágenes aumenta mucho. Por alguna razón desconocida, el tiempo en que crea las imágenes aumentaba en cada
	iteración.

3. Data generator:
	[Util]
	Este código contiene algunos efectos que se pueden aplicar a las imágenes usando la librería de 
	CV2. crea un conjunto de imágenes en una carpeta adicional, cómo son hartas imágenes toma su tiempo. 
	Por lo tanto, es mejor generar las imágenes en una carpeta aparte y leerlas cosa de usar de forma más 
	eficiente la instancia. Se usa CV2 porque es más rápido que el codigo de Tensorflow

4. eliminar carpetas 45 135, 225, 315:
	[Ya no es útil]
	Este codigo muestra como eliminar imágenes. Ya no es útil, se utilizó porque en algún momento de la memoria
	se sacaron imágenes en 45, 135, 225 y 315 grados. Sin embargo, usando las imágenes en 0, 90, 180 y 270
	grados era suficiente para construir una imágen panorámica
 
5. volver a descargar imagenes dañadas:
	[Incompleto]
	Este código está incompleto, pero algúnas imágenes no podían ser abiertas, probablemente porque hubo un 
	error en la descarga, por lo tanto, la finalidad de este archivo era crear un código que detectara las 
	imágenes mal descargadas y generar una nueva descarga.

6. Data resize 640x640:
	[Util]
	Este código permite ir recorriendo cada una de las imágenes que hay en la carpeta de IMÁGENES y achicarlas
	desde (640,640) a (224,224) y si hay una imágen panorámica transformarla de (640,640*4) a (224,224*224).
	
7. Detectar latitud longitud imagen:
	[Util]
	Las imágenes que se sacan de GSV tienen metadata (en un archivo .json). Este código permite recuperar la 
	longitud, latitud y la fecha de cada imágen. Lo que hace es recorrer cada una de las imágenes, a partir de
	su id y construye una base de datos con esa metadata.


