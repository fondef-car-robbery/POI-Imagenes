Codigos de la carpeta
1. concatenar imagenes - random santiago:
	[Util]
	Este codigo lee las im�genes separadas de 0, 90, 180 y 270 grados de la carpeta de IM�GENES (label_directory)
	y construye una im�gen panor�mica y luego la guardar en un directorio (guardar_en)

2. data_augmentation:
	[No utilizado]
	Este c�digo contiene algunos efectos que se pueden aplicar a las im�genes usando la librer�a de 
	tensorflow, sin embargo, no se recomienda utilizar porque el tiempo de procesamiento para generar las 
	im�genes aumenta mucho. Por alguna raz�n desconocida, el tiempo en que crea las im�genes aumentaba en cada
	iteraci�n.

3. Data generator:
	[Util]
	Este c�digo contiene algunos efectos que se pueden aplicar a las im�genes usando la librer�a de 
	CV2. crea un conjunto de im�genes en una carpeta adicional, c�mo son hartas im�genes toma su tiempo. 
	Por lo tanto, es mejor generar las im�genes en una carpeta aparte y leerlas cosa de usar de forma m�s 
	eficiente la instancia. Se usa CV2 porque es m�s r�pido que el codigo de Tensorflow

4. eliminar carpetas 45 135, 225, 315:
	[Ya no es �til]
	Este codigo muestra como eliminar im�genes. Ya no es �til, se utiliz� porque en alg�n momento de la memoria
	se sacaron im�genes en 45, 135, 225 y 315 grados. Sin embargo, usando las im�genes en 0, 90, 180 y 270
	grados era suficiente para construir una im�gen panor�mica
 
5. volver a descargar imagenes da�adas:
	[Incompleto]
	Este c�digo est� incompleto, pero alg�nas im�genes no pod�an ser abiertas, probablemente porque hubo un 
	error en la descarga, por lo tanto, la finalidad de este archivo era crear un c�digo que detectara las 
	im�genes mal descargadas y generar una nueva descarga.

6. Data resize 640x640:
	[Util]
	Este c�digo permite ir recorriendo cada una de las im�genes que hay en la carpeta de IM�GENES y achicarlas
	desde (640,640) a (224,224) y si hay una im�gen panor�mica transformarla de (640,640*4) a (224,224*224).
	
7. Detectar latitud longitud imagen:
	[Util]
	Las im�genes que se sacan de GSV tienen metadata (en un archivo .json). Este c�digo permite recuperar la 
	longitud, latitud y la fecha de cada im�gen. Lo que hace es recorrer cada una de las im�genes, a partir de
	su id y construye una base de datos con esa metadata.


