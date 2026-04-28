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

Posteriormente e realizó la separación del conjunto de datos en train, test y validation usando la función `train_test_split` de scikit-learn en dos etapas; la división se hizo de forma que el 20% de los datos se fueron al conjunto de prueba y a partir del 80% restante se generó un conjunto de validación. Se estableció un valor fijo de `random_state = 42` para asegurar la reproducibilidad del experimento y también se utilizó el parámetro `stratify = y` para mantener una proporción similar de clases en los subconjuntos y evitar que la evaluación esté sesgada por una distribución desigual entre sarcasmo y no sarcasmo. Como resultado final se obtuvieron 20,033 ejemplos para train, 2,862 para validation y 5,724 para test, lo cual permitió entrenar el modelo, monitorear su avance durante el ajuste y evaluar su desempeño sobre datos no vistos. 

Ya que se está trabajando con texto es necesario convertirlo en representación numérica para poder aplicar algoritmos de clasificación, por lo que se aplicó la capa `TextVectorization` de Keras, configurada con `max_tokens = 8000` y `output_sequence_length = 25`. La capa aprende un vocabulario a partir del conjunto de entrenamiento y transforma cada titular en una secuencia de enteros de longitud fija. Una vez adaptada la capa con los datos de entrenamiento se transformaron los conjuntos de train, validation y test, obteniendo matrices con dimensiones (20033, 25), (2862, 25) y (5724, 25) respectivamente. 

Finalmente se obtuvieron tres conjuntos secuenciales `X_train_vec`, `X_val_vec` y `X_test_vec` listos para ser utilizados como entrada del modelo; cada uno contiene los titulares transformados en secuencias enteras de longitud fija mediante `TextVectorization`. Al mismo tiempo, `y_train`, `y_val` y `y_test` contienen las etiquetas binarias asociadas a sarcasmo y no sarcasmo. 

# Referencia: artículo del estado del arte 

Para fundamentar la implementación del modelo se tomó como referencia el artículo [Advancing news headline sarcasm detection through hybrid neural networks](https://link.springer.com/article/10.1007/s10791-025-09877-8), publicado en *Discover Computing*. El trabajo propone un modelo híbrido de deep learning para la detección de sarcasmo en titulares de noticias utilizando una arquitectura basada en embeddings, capas convolucionales, MaxPooling, mecanismos de atención, BiLSTM y una salida densa con activación sigmoide para realizar una clasificación binaria entre titulares sarcásticos y no sarcásticos; el artículo justifica esta combinación porque las capas CNN ayudan a extraer patrones locales o expresiones cortas dentro del titular mientras que la BiLSTM permite capturar dependencias contextuales en ambas direcciones del texto.

Con base en dicha propuesta, este proyecto implementa una arquitectura más sencilla basada en **Embedding + Bidirectional LSTM + capas densas + Sigmoid**. Esta decisión se tomó para conservar la parte más relevante del enfoque secuencial del artículo, o sea el uso de una red bidireccional capaz de analizar el contexto anterior y posterior de cada palabra dentro del titular. No se replicó completamente la arquitectura propuesta en el artículo pero el modelo desarrollado mantiene la misma lógica general de trabajar con representaciones densas del texto y una salida sigmoide para resolver un problema de clasificación binaria.

El artículo también sirvió como base para la comparación final realizada en la novena iteración, donde se implementó una arquitectura alternativa más cercana al enfoque convolucional del trabajo de referencia (embedding + CNN/MaxPooling + capas densas + sigmoid), lo que permitió evaluar si la extracción de patrones locales mediante convoluciones podía superar al modelo con BiLSTM desarrollado durante el proyecto. Sin embargo los resultados obtenidos mostraron que la octava iteración (basada en BiLSTM) tuvo mejor desempeño en test que la arquitectura CNN/MaxPooling, por lo que se conservó como modelo final.

# Métricas 

Para evaluar el desempeño del modelo se utilizaron varias métricas tomando como referencia el artículo del estado del arte, el cual reporta principalmente `accuracy`, `precision`, `recall` y `F1-score`, ya que una sola no permite interpretar completamente el comportamiento del clasificador: en primer lugar se consideró el `accuracy` porque permite conocer el porcentaje general de titulares clasificados correctamente pero igual puede ocultar diferencias entre errores de tipo falso positivo y falso negativo, lo que justifica el uso de métricas adicionales; se utilizó `precision` para medir qué tan confiables son las predicciones positivas del modelo (cuántos de los titulares clasificados como sarcásticos sí lo eran) y también `recall` para medir cuántos de los titulares sarcásticos reales fueron detectados correctamente. El recall fue especialmente importante para el proyecto ya que uno de los objetivos principales era reducir los falsos negativos y evitar que titulares sarcásticos pasaran como no sarcásticos.

El `F1-score` se utilizó como métrica de equilibrio entre `precision` y `recall` ya que combina ambas en un solo valor. Fue útil para comparar iteraciones de forma más justa especialmente cuando un ajuste aumentaba la precisión pero disminuía el recall o viceversa. 

Finalmente en las últimas iteraciones se añadieron `ROC-AUC` y `PR-AUC`. El `ROC-AUC` permite evaluar la capacidad general del modelo para separar titulares sarcásticos y no sarcásticos considerando distintos umbrales de decisión, y el `PR-AUC` para analizar directamente la relación entre precision y recall para la clase positiva (titulares sarcásticos). Estas métricas complementaron la evaluación basada en un único threshold y ayudaron a la selección del modelo final.

# Implementación del modelo
Para la implementación del modelo se utilizó un enfoque de deep learning para clasificación binaria de texto. La elección de una red neuronal profunda se justifica porque el sarcasmo no depende únicamente de la presencia aislada de ciertas palabras, sino también del orden en que aparecen, de las relaciones contextuales entre términos y de contrastes semánticos dentro de una secuencia corta, por lo que se optó por una arquitectura recurrente capaz de procesar el texto como secuencia y no como un conjunto de palabras independientes. El flujo completo de entrenamiento se implementó en `sarcasm.py`, mientras que la limpieza textual se encapsuló en `clean.py` para poder reutilizarla tanto en entrenamiento como en la interfaz.

La arquitectura base utilizada en las mejores iteraciones del modelo fue una red secuencial con una capa de entrada para secuencias de longitud fija, una capa `Embedding(input_dim=8000, output_dim=64)`, una capa `Bidirectional(LSTM(32))`, una capa `Dropout(0.4)`, una capa densa oculta de 32 neuronas con activación `ReLU`, una segunda capa `Dropout(0.4)` y una capa final `Dense(1, activation="sigmoid")`. 

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

Los resultados son levemente inferiores a los de validation pero es normal al evaluar sobre datos completamente no vistos; se trata de una reducció pequeña por lo que se intepreta que el modelo generaliza razonablemente bien y que no se ajustó únicamente al conjunto de train. Particularmente mantener un `F1-score` cercano a 0.85 en test es una señal positiva porque resume equitativamente la capacidad del modelo para detectar sarcasmo sin generar muchos falsos positivos.

La matriz de confusión en test fue

[[2591  406

435  2292]]

Esto indica que el modelo identificó correctamente 2591 titulares no sarcásticos y 2292 sarcásticos, clasificó incorrectamente 406 no sarcásticos como sarcásticos y 435 sarcásticos como no sarcásticos; prácticamente se puede decir que el modelo se equivoca en ambos sentidos pero nuevamente lo hace de forma relativamente equilibrada. Que los falsos negativos sean ligeramente más que los falsos positivos sugiere que todavía hay un pequeño margen para mejorar la detección de titulares sarcásticos reales pero no compromete el desempeño general pues no es una diferencia muy grande.

De forma general estas métricas indican que el modelo sí logró aprender patrones del sarcasmo en los titulares y que tuvo un buen desempeño para ser una primera implementación; el `accuracy` mayor al 85% y `F1-score` cercano a 0.85 en test indica que la arquitectura funciona bien para este problema, y como los resultados de validation y test son parecidos se puede decir que el modelo generaliza de forma adecuada y no solo memoriza los datos de entrenamiento. Sin embargo el comportamiento de `val_loss` durante el entrenamiento sugiere inicios de overfitting y todavía se puede mejorar ajustando algunos hiperparámetros por ejemplo el tamaño del embedding, el número de unidades de la LSTM, el valor de `Dropout` o la longitud máxima de las secuencias. 

# Refinamiento del modelo 
Posterior a la iteración inicial, se realizaron ajustes en los hiperparámetros para refinar el modelo y mejorar su capacidad de generalización, reduciendo las señales de overfitting que se apreciaron en el entrenamiento; en esa primera iteración se alcanzaron métricas sólidas tanto en validation como en test, sin embargo el `loss` en validation indica que el error empeoraba a medida que aumentaba el desempeño en train, sugiriendo que el modelo base sí tenía capacidad suficiente para aprender pero igualmente memorizaba en exceso, por lo que a partir de este punto se orientó el refinamiento a controlar la complejidad del modelo y estabiizar el entrenamiento para equilibrar `precision`, `recall` y `F1-score`. 

El ajuste consistió básicamente en modificar de forma progresiva los hiperparámetros como el tamaño del `embedding` para evitar overfitting, reducir `max_tokens` y `output_sequence_length` para simplificar la entrada textual y eliminar ruido, ajuste del `learning_rate` paara suavizar la actualización de pesos y estabilizar el entrenamiento, y un cambio de LSTM a GRU para evaluar si una unidad recurrente más ligera mantendría o empeoraría el desempeño con menor complejidad. 

Las iteraciones se compararon objetivamaente gracias al uso de las métricas `accuracy`, `precision`, `recall` y `F1-score`, esta última con mayor importancia en test ya que permite evaluar de manera equilibrada qué tan bien identifica el modelo los casos sarcásticos y qué tan confiables son sus predicciones. 

# Hallazgos relevantes

> **Nota**: durante el refinamiento del modelo se realizaron varias pruebas intermedias modificando hiperparámetros como el tamaño del vocabulario (`max_tokens`), la longitud de secuencia (`output_sequence_length`), el nivel de regularización (`Dropout`), la tasa de aprendizaje (`learning_rate`), el tipo de capa recurrente y el ajuste del umbral de decisión, pero no todas las iteraciones se documentaron de forma individual porque algunas representaron cambios menores o variantes muy cercanas entre sí y los resultados no modificaron de manera significativa el desempeño general del modelo. Por esto la documentación de iteraciones se enfoca únicamente en las iteraciones 2, 3, 5, 7, 8 y 9 pues reflejan los hallazgos más relevantes del proceso como el primer mejor balance del modelo, el efecto de aumentar la regularización, el impacto de ajustar la tasa de aprendizaje, la recuperación de contexto con una mayor longitud de secuencia, la mejora metodológica mediante métricas adicionales y ajuste de threshold y la comparación final contra una arquitectura CNN/MaxPooling basada en el artículo de referencia.

## Comparativa de los ajustes realizados 

<img width="1057" height="734" alt="cambiosxiteracion" src="https://github.com/user-attachments/assets/f451210c-8d2d-47f1-86e7-193045a30489" />

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

con el objetivo de construir una red más pequeña y controlada que no tienda al overfitting, además de adaptarse mejor a titulares de noticias breves con el uso de solo 8000 tokens en lugar de 10000 evitando que aprenda detalles innecesarios. Los resultados indican que el ajuste fue exitoso pues aun cuando `precision` bajó ligeramente frente a la primera iteración, el `recall` subió considerablemente en especial en test (0.8812) indicando que el modelo detectó mejor los casos reales de sarcasmo y dejó pasar menos positivos, por lo que el `F1-score` en test subió a 0.8547 y fue el mejor valor de todas las iteraciones. En general la reducción moderada de complejidad combinada con una regularización equilibrada logró mejorar la sensibilidad del modelo sin penalizar mucho su precisión.

## Iteración 3

En esta iteración buscó regularizar aún más el modelo con la intención de disminuir más la complejidad y suavizar el aprendizaje. Estos fueron los ajustes en los hiperparámetros

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.4 --> **0.5**
- ***max_tokens:*** 8000 --> **5000**
- ***output_sequence_length:*** 25 --> **20**  
- ***patience:*** 2
- ***learning_rate:*** 0.0003

Se aumentó el `Dropout` de 0.4 a 0.5, se redujo el vocabulario máximo de 8000 a 5000 tokens y se disminuyó la longitud de secuencia de 25 a 20 palabras para evitar que el modelo aprendiera detalles demasiado específicos del conjunto de entrenamiento y obligarlo a concentrarse en patrones más generales del texto; esta regularización resultó demasiado fuerte pues aunque el modelo obtuvo una mayor precisión, su recall disminuyó implicando que fue más cuidadoso al clasificar un titular como sarcástico pero dejó escapar más casos reales de sarcasmo, o sea el modelo cometió menos falsos positivos pero aumentó los falsos negativos. Aunque esta iteración ayudó a controlar la complejidad, no fue la mejor opción para el objetivo principal del proyecto, ya que es muy importante detectar correctamente los titulares sarcásticos (priorizar true positives). 

## Iteración 5

En esta iteración se conservó la arquitectura base pero se redujo el `learning_rate` de 0.0005 a 0.0003 para hacer que el aprendizaje fuera más gradual y evitar actualizaciones demasiado grandes en los pesos del modelo. 

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.4
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 
- ***patience:*** 2
- ***learning_rate:*** 0.0005 --> **0.0003**

Esta modificación produjo una ligera mejora respecto a la iteración 4 pues en test se alcanzó un `accuracy` de 0.8564, `precision` de 0.8642, `recall` de 0.8287 y `F1-score` de 0.8461; aunque el `F1-score` subió frente a la iteración 4 el recall siguió siendo menor que el de la iteración 2, demostrando que el modelo sigue siendo más conservador al detectar sarcasmo. Se encontró que reducir el learning rate ayudó a estabilizar el entrenamiento pero no fue suficiente para obtener el mejor balance entre precision y recall. 

## Iteración 7

En la iteración 7 se regresó a la arquitectura con `BiLSTM`, se ajustó el `Dropout` de 0.4 a 0.35 y se aumentó nuevamente `output_sequence_length` de 20 a 25 para permitir que el modelo conserve más información del titular y tenga un poco más de capacidad de aprendizaje, pues se observó que las iteraciones con secuencia de 20 palabras tendían a perder recall. 

- ***Embedding:*** 64
- ***BiLSTM:*** 32
- ***Dense:*** 32 
- ***Dropout:*** 0.4 --> **0.35**
- ***max_tokens:*** 8000
- ***output_sequence_length:*** 20 --> **25**
- ***patience:*** 2
- ***learning_rate:*** 0.0005

La iteración logró uno de los mejores resultados después de la iteración 2 pues recuperó parte del balance entre precision y recall; reducir el `Dropout` permitió que el modelo no estuviera tan restringido durante el aprendizaje y la longitud de secuencia de 25 ayudó a conservar más contexto textual. No superó a la iteración 2 en test pero esta prueba confirmó que una regularización muy fuerte o una secuencia muy corta pueden limitar la detección de sarcasmo.

## Iteración 8 

La iteración 8 se enfocó en mejorar la forma de evaluación del modelo pues en las iteraciones anteriores el análisis se basaba principalmente en métricas como `accuracy`, `precision`, `recall` y `F1-score` y usaaba un threshold fijo de 0.5 para convertir las probabilidades en clases, por lo que en esta iteración se añadieron las métricas `ROC-AUC` y `PR-AUC` para observar mejor la capacidad de separación del modelo en varios puntos de decisión.

También se implementó un ajuste de threshold donde primero se calcularon las probabilidades del modelo sobre validation y después se probaron distintos umbrales (0.3, 0.4, 0.5, 0.6 y 0.7). El mejor threshold fue 0.3 porque obtuvo el mejor `F1-score` en validation; posteriormente ese mismo umbral se aplicó sobre test para evaluar si mejoraba el balance entre precision y recall. Los resultados de esta iteración fueron sólidos obteniendo un `F1-score` de 0.8651 en validation, `ROC-AUC` de 0.9406 y `PR-AUC` de 0.9359, lo que indica que el modelo tiene una buena capacidad para separar titulares sarcásticos y no sarcásticos. En test la iteración alcanzó `accuracy` de 0.8526, `precision` de 0.8201, `recall` de 0.8845 y `F1-score` de 0.8511. Aunque el F1-score quedó ligeramente por debajo de la iteración 2 la diferencia fue pequeña (0.8547 --> 0.8511).

La principal ventaja de la iteración 8 fue que detectó más casos positivos de sarcasmo; en test obtuvo 2412 true positives y solo 315 false negatives, mientras que la iteración 2 obtuvo 2403 true positives y 324 false negatives, o sea que la octava iteración detectó más titulares sarcásticos reales pero también generó más falsos positivos. Esta se considera una iteración importante y candidata a modelo final al priorizar la detección de sarcasmo sobre la precisión estricta de las predicciones positivas.

## Iteración 9 - Modelo comparativo basado en artículo de estado del arte 

Tomando como rerferencia el artíiculo de estado del arte se implementó la arquitectura `Embedding + CNN/MaxPooling + capas densas + Sigmoid` como prueba de mejora, ya que como los titulares son textos cortos pueden funcionar bien con una CNN pues aprende patrones locales de palabras como frases cortas o combinaciones típicas de sarcasmo (por ejemplo expresiones contradictorias, exageradas o irónicas); adicionalmente una CNN suele ser más ligera y rápida que una BiLSTM porque no procesa la secuencia paso a paso, más bien busca patrones relevantes en distintas posiciones del titular. Sin embargo no se esperaba una mejora relevante pues los resultados actuales reflejan que la BiLSTM ya captura el contexto en ambas direcciones, y aunnque la CNN puede mejorar la generalización/velocidad del modelo, igual podría perder una parte del contexto secuencial que sí aprende la BiLSTM. 

La novena iteración obtuvo un desempeño aceptable con un `F1-score` de 0.8529 en validation y 0.8358 en test, además de un `ROC-AUC` de 0.9215 y `PR-AUC` de 0.9102 en test, lo que demuestra que la CNN sí logró aprender patrones útiles en los titulares (combinaciones de palabras asociadas al sarcasmo), sin embargo su desempeño bajó al pasar a test, lo que indica una menor capacidad de generalización comparado al modelo anterior. La octava iteración obtuvo mejores resultados en test con `accuracy` de 0.8526, `recall` de 0.8845 y `F1-score` de 0.8511, superando a la iteración 9 (`accuracy` de 0.8398, `recall` de 0.8555 y `F1-score` de 0.8358); además la iteración 8 clasificó correcttamente más casos positivos de sarcasmo (2412 true positives) mientras que la CNN obtuvo 2333 true positives. Aunque la CNN tuvo una precisión muy similar generó más falsos negativos lo que es menos conveniente porque deja pasar más titulares sarcásticos sin detectarlos.


## Conclusiones generales
Aunque la iteración 2 obtuvo el F1-score más alto en test, la iteración 8 se seleccionó como modelo final porque ofrece una evaluación más completa y recupera más casos reales de sarcasmo, pues para el objetivo del proyecto es preferible reducir los falsos negativos y detectar más titulares sarcásticos aun cuando implique un ligero aumento de falsos positivos.

<h3>Gráficas modelo final (iteración 8)</h3>

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Accuracy por época</strong><br><br>
      <img src="https://github.com/user-attachments/assets/e19d3c9a-3bd4-4b22-92a2-59adde289fd9" width="520">
    </td>
    <td align="center" width="50%">
      <strong>Loss por época</strong><br><br>
      <img src="https://github.com/user-attachments/assets/c93a70e9-5177-4002-bcc9-5c77f1fcc926" width="520">
    </td>
  </tr>

  <tr>
    <td colspan="2">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <strong>Curva ROC</strong><br><br>
      <img src="https://github.com/user-attachments/assets/b3e4ab55-d37e-40a8-810a-67897c5bed64" width="480">
    </td>
    <td align="center" width="50%">
      <strong>Curva Precision-Recall</strong><br><br>
      <img src="https://github.com/user-attachments/assets/2f94d966-53c5-4a3c-abec-c35c8c27dc9e" width="480">
    </td>
  </tr>

  <tr>
    <td colspan="2">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center" colspan="2">
      <strong>Matriz de confusión</strong><br><br>
      <img src="https://github.com/user-attachments/assets/e980f4d8-8d71-4a52-8e65-73fb76d2b050" width="460">
    </td>
  </tr>
</table>

### Interpretación de las gráficas 

En la gráfica de accuracy por época se observa que el `accuracy` de entrenamiento aumenta de forma constante pasando aproximadamente de 0.76 a 0.95, pero el `accuracy` de validation se mantiene casi estable alrededor de 0.86 y después disminuye ligeramente. lo que indica que el modelo sigue aprendiendo muy bien los datos de entrenamiento pero la mejora ya no se refleja en validation, por lo que comienza a aparecer una señal de overfitting después de las primeras épocas.

La gráfica de loss por época confirma esta misma tendencia, pues el `loss` de entrenamiento disminuye de forma continua mientras que el `loss` de validation baja al inicio, alcanza su mejor punto cerca de la época 2 y después vuelve a subir; esto da a entender que el modelo empieza a ajustarse demasiado a los datos de entrenamiento or lo que el uso de `EarlyStopping` fue adecuado para recuperar los mejores pesos y evitar conservar una versión más sobreentrenada del modelo.

La matriz de confusión muestra que el modelo clasificó correctamente 2468 titulares no sarcásticos y 2412 titulares sarcásticos, también cometió 529 falsos positivos y 315 falsos negativos, lo que apoya la decisión de usar esta iteración como modelo final pues logra recuperar una gran cantidad de casos positivos de sarcasmo y mantiene relativamente bajos los falsos negativos.

La curva Precision-Recall obtuvo un `PR-AUC` de 0.9235 lo que indica un buen balance entre precision y recall en distintos umbrales de decisión; esta gráfica es especialmente relevante para el proyecto porque permite analizar el comportamiento del modelo al detectar la clase positiva (titulares sarcásticos). El valor alto de PR-AUC muestra que el modelo conserva un desempeño sólido aun cuando el threshold de clasificación cambia.

Finalmente la curva ROC obtuvo un `ROC-AUC` de 0.9304, se mantiene claramente por encima de la línea del modelo aleatorio indicando que el modelo tiene una buena capacidad para separar titulares sarcásticos y no sarcásticos. Las gráficas muestran que la Iteración 8 tiene un buen desempeño general aunque con señales de overfitting controladas mediante early stopping y ajuste del threshold.

---

### Referencias

1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using News Headlines Dataset." AI Open (2023).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
