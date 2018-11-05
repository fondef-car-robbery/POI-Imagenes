1. Text_mining_PI: 
	[Util]
	Este código permite hacer text mining sobre el nombre de cada punto de interés en la base de datos de PI y 
	de esta forma obtener una idéa que es lo que contiene cada subcategoría de la base de datos de PI
	
2. Conteo PI por subcategorias - Random Santiago:
	[Util]
	Este código realiza una caracterización de PI para cada (x,y) que está en la base de datos Random Santiago.
	se usa la función buscar_rango() para hace run subset de los PI que están cerca de (x,y). Esto aumentó 
	considerablemente la eficiencia del código en la caracterización disminuyendo el tiempo de dias a horas.
	Usa como input la base de datos en bruto sacada con web scraping llamada "data" y otra llamada "example"
	que tiene los x,y. El producto que genera la base es uno que se llama prose_PI.

3. Conteo PI por subcategorias - Random Santiago:
	[Util]
	Es lo mismo que (2.) pero usar la base de datos de la aseguradora. Aunque no tiene implementada la lógica 
	que mejora la eficiencia en el conteó de PI.
	
	