1. modelos_nopano:
	[Utilizado]
	Este c�digo tiene todos los modelos para im�genes no pano. Funciones que generan los modelos de CNN, guardan
	los pesos, guardan las m�tricas de desempe�o final, guardan el desempe�o del accuracy en todos los epoch como
	metada y como gr�fico, adem�s de llamar a las funciones de carga de datos en train y test y finalmente, 
	c�digos para inicializar el entrenamiento

2. modelos_pano:
	[Utilizado]
	Similar a (1.) pero cambia el tipo de input porque esta es una red con cuatro inputs

3. modelo_pano_bigimage:
	[Utilizado]
	Similar a (1.) pero cambia el tipo de input porque esta es una red con un �nico input que es la im�gen 
	panor�mica

Se separan las funciones de carga de datos por un tema de comodidad simplemente, si no, quedaban muchas lineas de c�digo