1. Text_mining_PI: 
	[Util]
	Este c�digo permite hacer text mining sobre el nombre de cada punto de inter�s en la base de datos de PI y 
	de esta forma obtener una id�a que es lo que contiene cada subcategor�a de la base de datos de PI
	
2. Conteo PI por subcategorias - Random Santiago:
	[Util]
	Este c�digo realiza una caracterizaci�n de PI para cada (x,y) que est� en la base de datos Random Santiago.
	se usa la funci�n buscar_rango() para hace run subset de los PI que est�n cerca de (x,y). Esto aument� 
	considerablemente la eficiencia del c�digo en la caracterizaci�n disminuyendo el tiempo de dias a horas.
	Usa como input la base de datos en bruto sacada con web scraping llamada "data" y otra llamada "example"
	que tiene los x,y. El producto que genera la base es uno que se llama prose_PI.

3. Conteo PI por subcategorias - Random Santiago:
	[Util]
	Es lo mismo que (2.) pero usar la base de datos de la aseguradora. Aunque no tiene implementada la l�gica 
	que mejora la eficiencia en el conte� de PI.
	
	