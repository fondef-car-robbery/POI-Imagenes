# Puntos-de-Interes

CARPETAS:
1. Caracterizacion puntos de interes:
	En esta carpeta se encuentran los códigos que transforman la base de datos de puntos de interés a una
	caracterización de lugares de Santiago en función de puntos de interés y un código para etender que 
	es lo que hay en la base de datos de puntos de interés
2. Filtros DB:
	En esta carpeta están los códigos que filtran los datos
3. Modelos:
	En esta carpeta están todos los modelos, funciones de carga de bases.
4. Procesos de imagenes:
	En esta carpeta están los códigos para procesar, extraer y pre-procesar imágenes

CODIGOS:
1. Generador de grilla y etiquetas - 13sep:
	Este código lee la base de datos de la aseguradora y construye una grilla de zonas de robo y no robo,
	luego esa grilla se subdivide para obtener un detalle de los lugares donde ha habido robo y donde no.	
	Finalmente, se crea un archivo que contiene las etiquetas de si una coordenada está en una zona de riesgo,
	una etiqueta de la cantidad de robos por zona, si es un lugar de robo, la frecuencia de robo.

2. puntoint_a_xy_geopy:
	Este código se utilizó para usar la API de google que permite convertir una dirección en una coordenada
	de longitud y latitud. 
3. 