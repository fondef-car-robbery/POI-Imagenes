1. Cargar_imagenes_v5_nopano:
	Este codigo es para cargar las im�genes y los labels en formato no panor�mico

2. Cargar_imagenes_pano_bigimage_12sep:
	Este codigo es para cargar las im�genes y los labels en formato panor�mico, como una im�gen de
	(224,224*4). Se utiliza este modelo para un input �nico que es la impagen panor�mica

3. Cargar_imagenes_pano_12sep:
	Este codigo es para cargar las im�genes y los labels en formato no panor�mico en un formato donde hay
	"cuatro inputs en paralelo" de im�genes en (224,224), pero al estar en "paralelo" es como una im�gen 
	panor�mica. Se usa esta carga de datos para el modelo que tiene 4 inputs

Todos los c�digos son muy similares.
