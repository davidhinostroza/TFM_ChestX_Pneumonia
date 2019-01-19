# Objetivo Proyecto TFM Deep Learning Rayos X del torax TPU.
Clasificar y detectar las enfermedades humanas a partir de las imagenes medicas de rayos X del torax como la Neumonia.
Para ello vamos a utilizar tecnologia google (unidad de procesamiento tensorial) TPU para comparar modelos binarios , Curvas Roc , matriz de confusión y comparar con un modelo pre-computado vs el modelo.

# Contexto

![texto alternativo](https://i.imgur.com/jZqpV51.png) 

### (imagen sustraida de Kaggle 


Tenemos images ilustrativos de radiografías de tórax en pacientes de neumonía. 
La radiografía de tórax normal (panel izquierdo) muestra los pulmones claros sin áreas opacas en la imagen del torax. 
La neumonía bacteriana (centro) exhibe una infección, en este caso en el lóbulo superior derecho (flechas blancas).
Mientras que la neumonía viral (derecha) se manifiesta con un patrón más difuso e "intersticial" en ambos pulmones.

Para ello construiremos un modelo de clasificación de imagen con Keras y tensorflow deep learning con tecnología TPU (unidad de procesamiento tensorial) donde utilizaremos modelo de tipo categórico con mediciones :


*   Matriz de confusión.
*   Curvas ROC AUC
*   Curvas ROC RECALL

Y se compara una red pre-computada para clasificación de imagen vs el modelo. 

# **Datos y Clasificación**

El conjunto de datos está organizado en 3 carpetas (train, test, val) y contiene subcarpetas para cada categoría de imagen (Pneumonia / Normal). 

Hay 5.863 imágenes de rayos X (JPEG) y 2 categorías (neumonía / normal).
Antes de cargar las imagenes tendre que etiquetarlas de la siguiente manera :							
- Tórax normal      = 0	
- Tórax Neumonia = 1

Fuente de datos : 

http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# **Arquitectura**


*   Google Colab Free TPU
*   Google Colab Free GPU
*   Python
*   Pandas
*   Keras

Primero ejecutar :

* El modelo propio TPU (TFM_Chest_TPU.ipynb)
* El modelo precomputado VGG16 (TFM_Chest_GPU_VGG16.ipynb)


