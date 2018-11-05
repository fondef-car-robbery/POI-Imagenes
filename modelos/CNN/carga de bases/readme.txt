1. Cargar_imagenes_v5_nopano:
	Este codigo es para cargar las imágenes y los labels en formato no panorámico

2. Cargar_imagenes_pano_bigimage_12sep:
	Este codigo es para cargar las imágenes y los labels en formato panorámico, como una imágen de
	(224,224*4). Se utiliza este modelo para un input único que es la impagen panorámica

3. Cargar_imagenes_pano_12sep:
	Este codigo es para cargar las imágenes y los labels en formato no panorámico en un formato donde hay
	"cuatro inputs en paralelo" de imágenes en (224,224), pero al estar en "paralelo" es como una imágen 
	panorámica. Se usa esta carga de datos para el modelo que tiene 4 inputs

Todos los códigos son muy similares.
