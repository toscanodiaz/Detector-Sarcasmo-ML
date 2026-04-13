# Detector-Sarcasmo-ML
Este proyecto implementa un flujo de preprocesamiento y preparación de datos para la detección de sarcasmo en titulares de noticias usando machine learning. A partir del dataset [News Headlines For Sarcasm Detection
](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) se realiza la limpieza del texto, la separación en conjuntos de entrenamiento y prueba, la vectorización con TF-IDF y el escalamiento de características, dejando los datos listos para entrenar y evaluar modelos de clasificación.

---

## Justificación del dataset
Se utilizará este dataset ya que tiene una estructura adecuada para un problema de clasificación de texto; contiene una columna con los titulares de noticias (`headline`) y una etiqueta binaria (`is_sarcastic`) que indica si el titular es sarcástico o no, facilitando trabajar directamente con técnicas de procesamiento de lenguaje natural al tener una entrada de texto clara y una salida bien definida para entrenar modelos supervisados. De la mano con esto también se eligió el dataset pues el objetivo es aplicar métodos vistos en clase a un problema real de clasificación, en particular el uso de `TF-IDF (Term frequency - Inverse document frequency)` tiene sentido en este contexto puees es un método para representar texto numéricamente según la importancia de las palabras dentro de un conjunto de documentos. Como el proyecto trata de detectar sarcasmo a partir de titulares, `TF-IDF` permite transformar el texto en vectores para usarlos en algoritmos de machine learning y conservar información relevante sobre la frecuencia e importancia de los términos.

## Descripción del preprocesado
El dataset está compuesto por titulares de noticias y una etiqueta `is_sarcastic` donde 1 representa que el texto es sarcástico y 0 que no lo es; también contiene las columnas `headline` con el texto a analizar y la columna `article_link` con los URLs que llevan a cada artículo. Esta última columna no fue utilizada para el preprocesamiento pues el objetivo es detectar el sarcasmo únicamente a partir de los titulares, por lo que no es una característica relevante para el análisis. 

Como primer paso se cargó el archivo en formato JSON utilizando pandas y posteriormente se realizó una inspección inicial del dataset para conocer sus dimensiones, verificar los nombres de las columnas, identificar valores nulos en caso de existir y revisar la distribución de las clases para confirmar que la variable de entrada sería el texto del titular `headline` y la variable de salida la clase binaria `is_sarcastic`. 

Después se definieron las variables del problema siendo `x` la variable independiente (los headlines) y `y` la variable dependiente (`is_sarcastic`) y comenzó el preprocesamiento, el cual consistió en limpiar y normalizar cada titular mediante varias transformaciones; todo el texto se convirtió a minúsculas para evitar confusión entre palabras iguales por diferenciar entre mayúsculas y minúsculas, también se eliminaron enlaces o URLs ya que no aportan información semántica relevante para la detección de sarcasmo en este contexto. Igualmente se removieron los números y los signos de puntuación porque no contribuyen a la interpretación de los titulares, además de que su eliminación ayuda a quitar el ruido en los datos. Para terminar se quitaron espacios innecesarios al inicio o al final de cada cadena de texto para obtener una versión homogénea de los titulares.

Posteriormente e realizó la separación del conjunto de datos en train y test usando la función `train_test_split` de scikit-learn; la división se hizo de forma que el 80% de los datos se van al conjunto de entrenamiento y el 20% al conjunto de prueba. Se estableció un valor fijo de `random_state = 42` para asegurar la reproducibilidad del experimento y también se utilizó el parámetro `stratify = y` para que las clases estén distribuídas uniformemente en ambos subconjuntos y evitar riesgo de overfitting o sesgos en el entrenamiento o evaluación del modelo.

Ya que se está trabajando con texto es necesario convertirlo en representación numérica para poder aplicar algoritmos de clasificación, por lo que se aplicó `TF-IDF` mediante la clase `TfidfVectorizer` que transforma los titulares en vectores numéricos que representan la importancia de las palabras dentro del documento y el corpus entero; dentro de esta vectorización se utilizó `stop_words='english'` para que el vectorizador ignore palabrwas muy comunes en inglés que no aportan nada en la distinción de clases, se definió el vocabulario a un máximo de 5000 características con `max_features=5000` y se usó `ngram_range=(1,2))` para capturar combinaciones de dos palabras y no solo palabras individuales, lo que aporta contexto para la detección de sarcasmo. 

Finalmente se aplicó un escalamiento en los vectorers resultantes con `MaxAbsScaler` que ajusta los valores sin afectar la estructura dispersa de los datos. El escalamiento se ajustó únicamente usando los datos en train y luego se aplicó a ambos para evitar fuga de información de test a train. 

Se obtuvo un dataset homogéneo y listo para entrenar con modelos de machine learning con cuatro estructuras principales: `X_train_scaled` y `X_test_scaled` que contienen los vectores escalados de los titulares para train y test, y `y_train` y `y_test` que contienen las etiquetas binarias de sarcasmo/no sarcasmo. 


---

### Referencias

1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using News Headlines Dataset." AI Open (2023).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
