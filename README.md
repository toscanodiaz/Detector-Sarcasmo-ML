# Detector-Sarcasmo-ML
Este proyecto implementa un flujo de preprocesamiento y preparación de datos para la detección de sarcasmo en titulares de noticias usando deep learning. A partir del dataset [News Headlines For Sarcasm Detection
](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) se realiza la limpieza del texto, la separación en conjuntos de entrenamiento, prueba y validación, la vectorización secuencial con TextVectorization y el entrenamiento de una red neuronal recurrente para clasificar titulares sarcásticos y no sarcásticos.

---

# Justificación del dataset
Se utilizará este dataset ya que tiene una estructura adecuada para un problema de clasificación de texto; contiene una columna con los titulares de noticias (`headline`) y una etiqueta binaria (`is_sarcastic`) que indica si el titular es sarcástico o no, facilitando trabajar directamente con técnicas de procesamiento de lenguaje natural al tener una entrada de texto clara y una salida bien definida para entrenar modelos supervisados. De la mano con esto también se eligió el dataset pues objetivo del proyecto es aplicar un modelo de deep learning a un problema real de clasificación de texto; los titulares de noticias son útiles porque acostumbran ser frases cortas, directas y con suficiente carga semántica como para que el modelo pueda aprender patrones asociados al sarcasmo. Como los headlines concentran la información en pocas palabras (a diferencia de otros textos más largos o ambiguos), son adecuados para trabajar con arquitecturas secuenciales. 

# Descripción del preprocesado
El dataset está compuesto por titulares de noticias y una etiqueta `is_sarcastic` donde 1 representa que el texto es sarcástico y 0 que no lo es; también contiene las columnas `headline` con el texto a analizar y la columna `article_link` con los URLs que llevan a cada artículo. Esta última columna no fue utilizada para el preprocesamiento pues el objetivo es detectar el sarcasmo únicamente a partir de los titulares, por lo que no es una característica relevante para el análisis. 

Como primer paso se cargó el archivo en formato JSON utilizando pandas y posteriormente se realizó una inspección inicial del dataset para conocer sus dimensiones, verificar los nombres de las columnas, identificar valores nulos en caso de existir y revisar la distribución de las clases para confirmar que la variable de entrada sería el texto del titular `headline` y la variable de salida la clase binaria `is_sarcastic`. 

Después se definieron las variables del problema siendo `x` la variable independiente (los headlines) y `y` la variable dependiente (`is_sarcastic`) y comenzó el preprocesamiento, el cual consistió en limpiar y normalizar cada titular mediante la función `limpiar_texto` implementada en el archivo `clean.py`; todo el texto se convirtió a minúsculas para evitar confusión entre palabras iguales por diferenciar entre mayúsculas y minúsculas, también se eliminaron enlaces o URLs ya que no aportan información semántica relevante para la detección de sarcasmo en este contexto. Igualmente se removieron los números y los signos de puntuación porque no contribuyen a la interpretación de los titulares, además de que su eliminación ayuda a quitar el ruido en los datos. Para terminar se quitaron espacios innecesarios al inicio o al final de cada cadena de texto para obtener una versión homogénea de los titulares.

Posteriormente e realizó la separación del conjunto de datos en train, test y validation usando la función `train_test_split` de scikit-learn en dos etapas; la división se hizo de forma que el 20% de los datos se fueron al conjunto de prueba y a partir del 80% restante se generó un conjunto de validación. Se estableció un valor fijo de `random_state = 42` para asegurar la reproducibilidad del experimento y también se utilizó el parámetro `stratify = y` para que las clases estén distribuídas uniformemente en los subconjuntos y evitar riesgo de overfitting o sesgos en el entrenamiento o evaluación del modelo. Como resultado final se obtuvieron 20,033 ejemplos para train, 2,862 para validation y 5,724 para test, lo cual permitió entrenar el modelo, monitorear su avance durante el ajuste y evaluar su desempeño sobre datos no vistos. 

Ya que se está trabajando con texto es necesario convertirlo en representación numérica para poder aplicar algoritmos de clasificación, por lo que se aplicó la capa `TextVectorization` de Keras, configurada con `max_tokens = 8000` y `output_sequence_length = 25`. La capa aprende un vocabulario a partir del conjunto de entrenamiento y transforma cada titular en una secuencia de enteros de longitud fija. Una vez adaptada la capa con los datos de entrenamiento se transformaron los conjuntos de train, validation y test, obteniendo matrices con dimensiones (20033, 30), (2862, 30) y (5724, 30) respectivamente. 

Finalmente se obtuvieron tres conjuntos secuenciales `X_train_vec`, `X_val_vec` y `X_test_vec` listos para ser utilizados como entrada del modelo; cada uno contiene los titulares transformados en secuencias enteras de longitud fija mediante `TextVectorization`. Al mismo tiempo, `y_train`, `y_val` y `y_test` contienen las etiquetas binarias asociadas a sarcasmo y no sarcasmo. 

# Implementación del modelo
Para la implementación del modelo se utilizó un enfoque de deep learning para clasificación binaria de texto. La elección de una red neuronal profunda se justifica porque el sarcasmo no depende únicamente de la presencia aislada de ciertas palabras, sino también del orden en que aparecen, de las relaciones contextuales entre términos y de contrastes semánticos dentro de una secuencia corta, por lo que se optó por una arquitectura recurrente capaz de procesar el texto como secuencia y no como un conjunto de palabras independientes. El flujo completo de entrenamiento se implementó en `sarcasm.py`, mientras que la limpieza textual se encapsuló en `clean.py` para poder reutilizarla tanto en entrenamiento como en la interfaz.

La arquitectura final del modelo fue una red secuencial con una capa de entrada para secuencias de longitud fija, una capa `Embedding(input_dim=8000, output_dim=64)`, una capa `Bidirectional(LSTM(32))`, una capa `Dropout(0.4)`, una capa densa oculta de 64 neuronas con activación `ReLU`, una segunda capa `Dropout(0.4)` y una capa final `Dense(1, activation="sigmoid")`. 

La capa Embedding se utilizó porque una red neuronal no trabaja directamente con texto crudo, sino con representaciones numéricas densas; cada token fue proyectado a un vector de dimensión 64 pues tienen suficiente capacidad para aprender relaciones semánticas entre palabras pero no incrementan de forma excesiva el número de parámetros.

La capa principal del modelo fue una Bidirectional LSTM con 32 unidades. Se eligió una LSTM porque este tipo de red está diseñada para trabajar con secuencias y capturar dependencias contextuales entre palabras; se usó en su versión bidireccional para que el modelo pudiera procesar el titular de izquierda a derecha y viceversa, lo cual es útil en la detección del sarcasmo ya que muchas veces la interpretación depende del contraste entre el inicio y el final del enunciado o de cómo una palabra cambia de significado según el contexto que se le da. El valor de 32 unidades brinda suficiente capacidad de modelado sin volver el entrenamiento innecesariamente costoso para el tamaño del dataset y la longitud de las secuencias.

Las capas `Dropout(0.4)` se añadieron como mecanismo de regularización para desactivar aleatoriamente una parte de las neuronas durante el entrenamiento y reducir el riesgo de overfitting, y con valor de 0.4 para frenar la memorización excesiva de patrones del conjunto de train sin ser tan alto para impedir el aprendizaje. Entre ambas capas de dropout se colocó una capa densa con 32 neuronas y activación ReLU para procesar la información aprendida por la LSTM y combinarla de forma más útil para la clasificación final. La salida se modeló con una única neurona y activación `sigmoid` porque el problema es binario y se requiere una salida entre 0 y 1 (interpretable como probabilidad de sarcasmo). 

El modelo se compiló con el optimizador `Adam`, la función de pérdida `binary_crossentropy` y la métrica `accuracy`. Se eligió Adam por su buen comportamiento general en redes neuronales, principalmente al trabajar con datos textuales y múltiples parámetros. La función `binary_crossentropy` es la más adecuada para una salida sigmoide en tareas de clasificación binaria pues penaliza más fuertemente las predicciones erróneas y permite entrenar el modelo en términos probabilísticos; también se configuró un entrenamiento de 10 épocas con tamaño `batch_size = 32` que es un valor estándar para equilibrar estabilidad del gradiente y eficiencia computacional.

Para controlar el overfitting se incorporó `EarlyStopping` monitoreando `val_loss`, con `patience = 2` y `restore_best_weights = True` buscando maximizar el desempeño en entrenamiento y conservvar la mejor capacidad de generalización sobre datos no vistos. Con `patience = 2` el entrenamiento puede agregar algunas épocas más si validation deja de mejorar momentáneamente pero sin prolongarse de forma innecesaria. También se restauran los mejores pesos para asegurar que el modelo final corresponda al punto de mejor desempeño en validation y no al de la última época entrenada. 

Finalmente el modelo entrenado se guardó en el archivo `modelo_dl.keras` y el vocabulario aprendido por `TextVectorization` se exportó a `vocabulario.txt` usando codificación UTF-8 y reconstruyendo exactamente la misma transformación de texto en la interfaz final implementada con Gradio; el usuario puede escribir un titular y analizarlo para obtener una predicción de sarcasmo o no sarcasmo junto con su probabilidad. 

# Evaluación inicial 
La evaluación inicial del modelo se realizó primero sobre el conjunto de validation para monitorear el entrenamiento y detectar el mejor punto de aprendizaje, y después sobre el conjunto de test para estimar la capacidad de generalización final sobre datos completamente no vistos. 

## Iteración 1

Se utilziaron los siguientes hiperparámetros, marcando la línea base del modelo: 

- ***Embedding:*** 128
- ***BiLSTM:*** 64
- ***Dense:*** 64 
- ***Dropout:*** 0.3
- ***max_tokens:*** 10000
- ***output_sequence_length:*** 30 
- ***patience:*** 3

Durante el entrenamiento se observó una mejora rápida en las métricas de train, donde `accuracy` pasó aproximadamente de 0.8062 en la primera época a 0.9719 en la cuarta, mientras que `loss` disminuyó de 0.4068 a 0.0733 indicando que la red fue capaz de aprender patrones de manera efectiva dentro del conjunto de train. Al mismo tiempo `val_loss` fue mejor en la primera época (0.3131) y empeoró progresivamente en las siguientes y `val_accuracy` dejó de mejorar después del inicio, lo que indica que el modelo comenzó a mostrar señales de overfitting a medida que avanzaban las épocas, por lo cual se justifica la incorporación de EarlyStopping.

### Validation
En el conjunto de validación el modelo obtuvo los siguientes resultados

- accuracy = 0.8651
- precision = 0.8574
- recall = 0.8599
- F1-score = 0.8586

El `accuracy` de 0.8651 significa que aproximadamente el 86.5% de los titulares de validation se clasificaron correctamente. Esta métrica es útil como visión general del desempeño pero no es suficiente por sí sola para evaluar un problema de clasificación binaria, por lo que también se analizaron precision, recall y F1-score.

La `precision` de 0.8574 indica que cerca del 85.7% de los titulares que el modelo clasificó como sarcásticos sí lo eran; con esto se observa que el modelo comete una cantidad moderada de falsos positivos pero mantiene una cantidad razonablemente alta de aciertos cuando etiqueta como sarcasmo. El `recall` de 0.8599 muestra que el modelo detectó alrededor del 86.0% de los casos sarcásticos reales, o sea la mayor parte de los positivos presentes en el conjunto de validación. El `F1-score` de 0.8586 combina precision y recall e indica que ambas métricas están bien equilibradas y que el modelo no está beneficiando a una y penalizando otra.

La matriz de confusión de validation fue 

[[1304  195

191  1172]]

Esto significa que el modelo clasificó correctamente 1304 titulares no sarcásticos y 1172 sarcásticos, cometió 195 falsos positivos y 191 falsos negativos; la cercanía entre ambos tipos de error indica que el desempeño del modelo en validation es bastante balanceado y que no existe bias hacia ninguna de las clases. El reporte por clase refleja valores cercanos entre la clase 0 y la clase 1 lo que sugiere que el modelo logra distinguir ambas categorías con un nivel de desempeño parecido.

### Test
En el conjunto de prueba, los resultados fueron

- accuracy = 0.8531
- precision = 0.8495
- recall = 0.8405
- F1-score = 0.8450

Los resultados son levemente inferiores a los de validation pero es normal al evaluar sobre datos completamente no vistos; se trata de una reducció pequeña por lo que se intepreta que el modelo generaliza razonablemente bien y que no se ajustó únicamente al conjunto de train. Particularmente mantener un F1-score cercano a 0.85 en test es una señal positiva porque resume equitativamente la capacidad del modelo para detectar sarcasmo sin generar muchos falsos positivos.

La matriz de confusión en test fue

[[2591  406

435  2292]]

Esto indica que el modelo identificó correctamente 2591 titulares no sarcásticos y 2292 sarcásticos, clasificó incorrectamente 406 no sarcásticos como sarcásticos y 435 sarcásticos como no sarcásticos; prácticamente se puede decir que el modelo se equivoca en ambos sentidos pero nuevamente lo hace de forma relativamente equilibrada. Que los falsos negativos sean ligeramente más que los falsos positivos sugiere que todavía hay un pequeño margen para mejorar la detección de titulares sarcásticos reales pero no compromete el desempeño general pues no es una diferencia muy grande.

De forma general estas métricas indican que el modelo sí logró aprender patrones del sarcasmo en los titulares y que tuvo un buen desempeño para ser una primera implementación; el `accuracy` mayor al 85% y `F1-score` cercano a 0.85 en test indica que la arquitectura funciona bien para este problema, y como los resultados de validation y test son parecidos se puede decir que el modelo generaliza de forma adecuada y no solo memoriza los datos de entrenamiento. Sin embargo el comportamiento de `val_loss` durante el entrenamiento sugiere inicios de overfitting y todavía se puede mejorar ajustando algunos hiperparámetros por ejemplo el tamaño del embedding, el número de unidades de la LSTM, el valor de `Dropout` o la longitud máxima de las secuencias. 

# Refinamiento del modelo 
Posterior a la iteración inicial, se realizaron ajustes en los hiperparámetros para refinar el modelo y mejorar su capacidad de generalización, reduciendo las señales de overfitting que se apreciaron en el entrenamiento; en esa primera iteración se alcanzaron métricas sólidas tanto en validation como en test, sin embargo el `loss` en validation indica que el error empeoraba a medida que aumentaba el desempeño en train, sugiriendo que el modelo base sí tenía capacidad suficiente para aprender pero igualmente memorizaba en exceso, por lo que a partir de este punto se orientó el refinamiento a controlar la complejidad del modelo y estabiizar el entrenamiento para equilibrar `precision`, `recall` y `F1-score`. 

El ajuste consistió básicamente en modificar de forma progresiva los hiperparámetros como el tamaño del `embedding` para evitar overfitting, reducir `max_tokens` y `output_sequence_length` para simplificar la entrada textual y eliminar ruido, ajuste del `learning_rate` paara suavizar la actualización de pesos y estabilizar el entrenamiento, y un cambio de LSTM a GRU para evaluar si una unidad recurrente más ligera mantendría o empeoraría el desempeño con menor complejidad. 

Las iteraciones se compararon objetivamaente gracias al uso de las métricas `accuracy`, `precision`, `recall` y `F1-score`, esta última con mayor importancia en test ya que permite evaluar de manera equilibrada qué tan bien identifica el modelo los casos sarcásticos y qué tan confiables son sus predicciones. 

## Comparativa de los ajustes realizados 

<img width="1920" height="1080" alt="cambiosxiteracion" src="https://github.com/user-attachments/assets/3ffcd27e-051a-4834-96ec-10bb3604b054" />

## Iteración 2

En esta iteración se redujo la complejidad del modelo con los siguientes ajustes  

- ***Embedding:*** 128 --> **64**
- ***BiLSTM:*** 64 --> **32**
- ***Dense:*** 64 --> **32**
- ***Dropout:*** 0.3 --> **0.4**
- ***max_tokens:*** 10000 --> **8000**
- ***output_sequence_length:*** 30 --> **25** 
- ***patience:*** 3 --> **2**
- ***learning_rate:*** **0.0005**

con el objetivo de construir una red más pequeña y controlada que no tienda al overfitting, además de adaptarse mejor a titulares de noticias breves con el uso de solo 8000 tokens en lugar de 10000 evitando que aprenda detalles innecesarios. Los resultados indican que el ajuste fue exitoso pues aun cuando la precision bajó ligeramente frente a la primera iteración, el recall subió considerablemente en especial en test (0.8812) indicando que el modelo detectó mejor los casos reales de sarcasmo y dejó pasar menos positivos, por lo que el F1-score en test subió a 0.8547 y fue el mejor valor de todas las iteraciones. En general la reducción moderada de complejidad combinada con una regularización equilibrada logró mejorar la sensibilidad del modelo sin penalizar mucho su precisión.

## Iteración 3 *****DEJAR

En esta iteración buscó regularizar aún más el modelo con la intención de disminuir más la complejidad y suavizar el aprendizaje. Estos fueron los ajustes en los hiperparámetros

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.4 --> **0.5**
- ***max_tokens:*** 8000 --> **5000**
- ***output_sequence_length:*** 25 --> **20**  
- ***patience:*** 2
- ***learning_rate:*** 0.0005

## Iteración 4

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.5 --> **0.45**
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 
- ***patience:*** 2
- ***learning_rate:*** 0.0005

## Iteración 5

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.45 --> **0.4**
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 
- ***patience:*** 2
- ***leanring_rate:*** 0.0005 --> **0.0003**

## Iteración 6 *****DEJAR

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.4
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 
- ***patience:*** 2
- ***learning_rate:*** 0.0005
- ***tf.keras.layers:*** LSTM(32) --> **GRU(32)**

## Iteración 7

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 4 --> **0.35**
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 --> **25**
- ***patience:*** 2
- ***learning_rate:*** 0.0005

## Iteración 8 *****DEJAR

— cambios —

- agregué métrica de curvas roc durante entrenamiento
- probé diferentes thresholds


---

### Referencias

1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using News Headlines Dataset." AI Open (2023).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
