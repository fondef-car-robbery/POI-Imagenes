1. modelos_nopano:
	[Utilizado]
	Este código tiene todos los modelos para imágenes no pano. Funciones que generan los modelos de CNN, guardan
	los pesos, guardan las métricas de desempeño final, guardan el desempeño del accuracy en todos los epoch como
	metada y como gráfico, además de llamar a las funciones de carga de datos en train y test y finalmente, 
	códigos para inicializar el entrenamiento

2. modelos_pano:
	[Utilizado]
	Similar a (1.) pero cambia el tipo de input porque esta es una red con cuatro inputs

3. modelo_pano_bigimage:
	[Utilizado]
	Similar a (1.) pero cambia el tipo de input porque esta es una red con un único input que es la imágen 
	panorámica

Se separan las funciones de carga de datos por un tema de comodidad simplemente, si no, quedaban muchas lineas de código