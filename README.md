# Percepcion_Actvidad_Procesamiento_IMG.

# 1. Descripción del script 

## 1.1. Instalación e Importación de Dependencias
Se realiza el llamado de las librerías y herramientas necesarias.
* **Importaciones estándar:** Carga librerías nativas de Python como `os` y `sys` que se usan para interactuar con el sistema operativo, `shutil` que se usa para mover archivos y `zipfile` para descomprimir.
* **Librerías de cálculo y visión:** Importa `numpy` para operaciones matemáticas, `matplotlib.pyplot` para crear las gráficas de entrenamiento y `cv2` (OpenCV), que se usa para la manipulación y análisis de imágenes.
* **Bloque try-except:** Es una medida de seguridad. Intenta importar `tensorflow` para la red neuronal y `ultralytics` para YOLO. Si el entorno no las tiene instaladas lanza un `ImportError`, y ejecuta automáticamente un comando `!pip install` para descargarlas de forma silenciosa.
* **Detección de entorno:** Verifica si el código se está ejecutando dentro de Google Colab (`"google.colab" in sys.modules`). Si es así, importa herramientas específicas de Colab, como la interfaz para subir archivos `files` y la función parcheada de OpenCV para mostrar imágenes sin colgar el navegador `cv2_imshow`.

## 1.2. Configuración y Creación de Carpetas Temporales
Se definen las rutas para procesar los datos y se definen las ubicaciones.
* **Rutas base:** Se define una ruta donde se creará una carpeta local `./dataset`.
* **Clases:** Establece las categorías que la red neuronal intentará aprender: `manzanas_rojas` y `otras_frutas`.
* **Creación de directorios:** Utiliza un bucle anidado junto con `os.makedirs` para crear automáticamente la estructura de carpetas, las carpetas `train` y `validation`, y dentro de cada una, subcarpetas para las dos clases. El argumento `exist_ok=True` evita errores si las carpetas ya existen de una ejecución anterior.

## 1.3. Función para Organizar Archivos Subidos
Se define una función para organizar archivos que se encarga de procesar lo que el usuario sube al entorno.
* **Descompresión:** Si detecta que el archivo termina en `.zip`, se extrae el contenido directamente en la carpeta de destino (`TRAIN_DIR` o `VAL_DIR`) y luego elimina el archivo `.zip`.

## 1.4. Carga de Imágenes al Entorno
Se solicita cargar los archivos con los datos al usuario.
* **Uso de `files.upload()`:** Utiliza `files.upload()` dos veces consecutivas: primero solicita el archivo comprimido en `.ZIP` con las imágenes para el entrenamiento y se guarda en el (`TRAIN`) y luego para el conjunto de validación (`VAL`).
* **Organización:** Por cada subida, invoca la función creada en el paso anterior con la que se organiza los archivos recién cargados.

## 1.5. Preparación de los Datasets para TensorFlow
Convierte los archivos de imagen crudos en tensores matemáticos que la red neuronal puede procesar.
* **Dimensiones y Lotes:** Define el tamaño estándar al que se redimensionarán todas las imágenes (en este caso a 150x150 píxeles) y la cantidad de imágenes que se procesarán a la vez antes de actualizar los pesos de la red (*Batch Size* de 32).
* **Carga automática:** Utiliza la poderosa función `image_dataset_from_directory` de Keras, la cual lee automáticamente la estructura de carpetas.
* **Optimización de rendimiento (Pipeline de datos):** Guarda las imágenes en la memoria RAM después de la primera época, haciendo que las épocas siguientes sean muchísimo más rápidas.
    * `.shuffle(1000)`: Mezcla aleatoriamente las imágenes para evitar que la red memorice el orden.
    * `.prefetch(AUTOTUNE)`: Permite que la CPU prepare el siguiente lote de imágenes mientras la GPU (si está activa) entrena el lote actual, eliminando cuellos de botella.

## 1.6. Construcción de la Red Neuronal Convolucional (CNN)
Se define la arquitectura del modelo utilizando Keras `models.Sequential`.
* **Rescaling:** Normaliza los píxeles. Pasa los valores de color de 0-255 a un rango de 0-1, lo cual estabiliza las matemáticas internas de la red.
* **Bloques Convolucionales (`Conv2D` + `MaxPooling2D`):** Hay tres de estos bloques. Las capas `Conv2D` aplican filtros para extraer patrones visuales como bordes, texturas, formas de manzanas. Las capas `MaxPooling2D` reducen el tamaño de la imagen, comprimiendo la información más importante para hacer el modelo más ligero y rápido.
* **Flatten:** Aplana el mapa de características 2D resultante en un vector para poder conectarlo a una red neuronal tradicional.
* **Capa Densa y Dropout:** Una capa densa de 128 neuronas procesa las características abstraídas. El `Dropout(0.5)` apaga aleatoriamente la mitad de las neuronas en cada paso, lo que obliga a la red a no depender de un solo patrón y evita el sobreajuste.
* **Salida (`softmax`):** La función `softmax` convierte la salida en porcentajes de probabilidad.
* **Compilación:** Se configura para usar el optimizador `adam` que ajusta inteligentemente la velocidad de aprendizaje y la función de pérdida.

## 1.7. Entrenamiento del Modelo
Encargado del aprendizaje profundo.
* **Ejecución:** Se llama a `model.fit()`, que entrega al dataset de entrenamiento y el de validación.
* **Configuración:** Se configura para 20 épocas (`EPOCHS`), lo que significa que el modelo revisará el conjunto de datos completo 20 veces, ajustando sus pesos internos en cada pasada para intentar reducir su porcentaje de error (*Loss*).
* **Historial:** El resultado del entrenamiento se guarda en la variable `history` para su análisis posterior.

## 1.8. Visualización de Rendimiento General
Extrae las métricas registradas en la variable `history` (creada en el paso anterior) para graficarlas de forma comprensible.
* **Estructura:** Utiliza `matplotlib` para crear un lienzo con dos subgráficos.
* **Gráfica 1 (Precisión/Accuracy):** Compara el porcentaje de aciertos en los datos de entrenamiento vs. los de validación a lo largo de las 20 épocas.
* **Gráfica 2 (Pérdida/Loss):** Muestra cómo disminuye el error matemático del modelo.
* **Análisis:** Esta visualización es la que permitió (en el análisis previo) detectar el comportamiento anómalo de memorización instantánea (curvas planas).

## 1.9. Funciones Auxiliares para Detección de Color (Rojo)
Prepara una herramienta de Visión por Computadora clásica que servirá como filtro en la Fase 2 cuando se haga uso de YOLO.
* **Función `es_color_rojo`:** Recibe un recorte de imagen donde YOLO detectó un objeto.
* **Conversión a HSV:** Utiliza `cv2.cvtColor` para pasar la imagen del modelo de color BGR (Azul, Verde, Rojo) a HSV (Matiz, Saturación, Valor).
* **Máscaras de color (`cv2.inRange`):** El color rojo en el espectro HSV es peculiar porque se encuentra en ambos extremos del cilindro (cerca del 0 y cerca del 180). Por eso se crean dos rangos y se suman (`+`), abarcando todos los tonos de rojo.
* **Cálculo de proporción:** Cuenta cuántos píxeles cumplieron con ser rojos (`cv2.countNonZero`) y lo divide por el total de píxeles del recorte. Si ese número supera 0.20 (es decir, el 20% del objeto detectado es de color rojo), entonces la función devuelve `True`.

---

# 2. Conclusiones y Recomendaciones

* **2.1. Validación de la Lógica Híbrida:** El diseño de combinar un modelo de detección de siluetas YOLO con un filtro de color clásico OpenCV es lo que permite detectar objetivos en escenarios controlados y claros.
* **2.2. Sobreajuste en la CNN:** La Red Neuronal Convolucional depende de los datos de origen, por lo cual es necesario tener en cuenta el tipo de imágenes que se van a cargar en las carpetas `train` y `validation` para asegurar una correcta separación de los datos. Una buena opción para mejorar los resultados es aumentar el número de imágenes y también se podrían aplicar técnicas como rotación, zoom y cambios de luz para mejorar el aprendizaje del modelo.
* **2.3. Limitaciones de las Reglas Estrictas:** El filtro HSV actual es propenso a errores por iluminación o reflejos (como por ejemplo el agua en las manzanas o cortes en la fruta), lo cual se evidenció en los resultados obtenidos al ejecutar el script.
* **2.4. Contexto de la Imagen:** Hay que tener en cuenta que una regla basada solamente en el porcentaje de un color falla en interpretar el contexto general de la imagen.
* **2.5. Optimización:** Una forma de mejorar el sistema del script (el cual está dividido en dos enfoques) sería agregar un etiquetado con cajas delimitadoras específicas para cada fruta y reentrenar el modelo YOLOv8 directamente sobre estas clases personalizadas. Al realizar esto se elimina la debilidad que tiene usar filtros de color de OpenCV y la CNN inicial, generando un modelo de detección mucho más robusto.
