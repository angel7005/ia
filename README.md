
#### Contents

* [Introduccion](#Introducción)
* [Solución](#Solución)
* [Preparación y depuración de los datos](#Preparación-y-depuración-de-los-datos)
* [Datos de entrenamiento y pruebas](#Datos-de-entrenamiento-y-pruebas)
* [Seleccionar modelo y ajustar parámetros](#Seleccionar-modelo-y-ajustar-parámetros)
	* [RandomForest](#RandomForest)
	* [Gradiente Boosting](#Gradiente-Boosting)
	* [MLPClassifier](#MLPClassifier)
	* [Regresion logistica](#Regresion-logistica)
* [Entrenar modelo](#Entrenar-modelo)
* [Evaluar rendimiento](#Evaluar-rendimiento)
* [Hacer predicciones](#Hacer-predicciones)
* [Comparaciones entre modelos](#Comparaciones-entre-modelos)
* [Conclusiones](#Conclusiones)
* [Recomendaciones](#Recomendaciones)
* [Bibliografía](#Bibliografía)


## Introducción

Para una familiarizacion con el problema a resolver leer el documento Customer_Churn.md, en la carpeta actual. tambien puede ser encontrado el archivo archive.zip con el juego de datos. En el se puede observar el cliente, las variables predictoras y la variable objetico(Churn). Un poco más de 7000 instancias de clientes.

Entre las variables predictoras hay variables de tipo boolean, categoricas, enteras y continúas. Menos de 20 variables.

La variable objetivo (churn) tiene dos valores válidos: No, Yes. Nos interesa predecir a los clientes con valor 'Yes', que abandonarán la empresa.
De la observación de los datos proporcionados se observa una muestra sesgada o con desbalance a favor de la clase negativa 'No', con mas del 75%. 

Esto puede dificultar el aprendisaje del modelo al contar con pocos clientes con  valor 'Yes' en la variable objetivo, en comparacion 
con el valor 'No' que es la clase mayoritaria

La selección de variables con ayuda del especialista puede ayudar a depurar la muestra de clientes suministrada,
comenzando por los falsos positivos y próximo a la frontera o al umbral de clasificación según el modelo

Encontrar las variables más relevantes en la clasificación de los clientes, ayudará objetivamente a la empresa a saber donde puede invertir su dinero para retener al cliente. 

La solución pasa por tener en cuenta el problema del desbalance de la muestra, predecir si un cliente abandonara la empresa y cuales son las variables más importantes que influyen en que el cliente tome esa decisión.


## Solución

Dado que el problema se perfila como un problema de clasificación binaria con aprendizaje, los algoritmos candidatos
son RandomForest, GradientBoosting, MLPClassifier, SVC, LogisticRegression. 

Para lo que se crea un cuaderno por cada modelo y uno para la comparación.

Preliminarmente se hicieron algunas corridas con los datos de pruebas proporcionados y se compararon las métricas 
resultados de diferentes algoritmos (RandomForest, GradientBoosting, MLPClassifier, LogisticRegression, SVM). Se realizó una búsqueda en 
google para la recomendación de algoritmos alternativo al RandomForest y estos fueron los ganadores. El próximo paso sería
actualizarme en algunos y conocer en otros de las ventajas y desventajas.

Retomando el problema del desbalance de la muestra se valora entre las técnicas propuestas el remuestreo y ajuste de los parámetros del algoritmo dando mas peso a la clase minoritaria ('Yes').

El proyecto se desarrolla con python como lenguaje de programación y las bibliotecas sklearn de scikit-learn. Principalmente con el 
algoritmo RandomForest y la clase GridSearchCV que es una técnica de ajuste de hiperplanos para econtrar la combinación óptima de los parámetros del algoritmo y evitar el OverFitting o sobre ajuste utilizando la validación cruzada.


## Preparación y depuración de los datos
Al cargarse los datos se encuentran valores omitidos en la penúltima columna "TotalCharge". 

* Cargos totales vacíos, los pongo a 0.0, los visualice y coincide con que:
	* a) La variable tenure es "0",  el número de meses(tenure) que el cliente  ha estado en la empresa es cero.
	* b) La variable Churn tienen valor "No", Por lo que se deduce que no ha tenido tiempo de cobrar la mensualidad o no han podido pagar aun.
	* c) Dado que la muestra esta sesgada a la clase 'No', pudiera simplemente no incluirla en el entrenamiento e incluirla en la submuestra de prueba. Pero asocie a estos clientes como clientes recien incorporados al sistema.
    
Se observan muchas variables categoricas, con las que los modelos no pueden trabajar.
       Se tranforman en valores numéricas.

## Datos de entrenamiento y pruebas
Los datos de separaron los primeros 90% para entrenamiento y el ultimo 10% para prueba. Ahora el 90% de entrenamiento se le dio a GridSearchCV, utilizando validación cruzada con 5, 7 y 9 particiones sobre el que se ajustaron los parámetros del modelo

## Seleccionar modelo y ajustar parámetros
Se crea un cuaderno por cada modelo donde se van a optimizar sus parámetros. Utilizando GridSearchCV. Para luego comparar sus resultados.

### RandomForest
Cuaderno ns-3.1.1-modelo-RandomForest.ipynb

### Gradiente Boosting
cuaderno ns-3.1.2-modelo-GradientBoosting.ipynb

### MLPClassifier
Cuaderno ns-3.1.3-modelo-MLPClassifier.ipynb

### Soporte Vector Machine
Cuaderno ns-3.1.4-modelo-SVC.ipynb

### Regresion logistica
Cuaderno ns-3.1.5-modelo-LogisticRegression.ipynb

## Entrenar modelo
En el cuaderno  ns-4-comparacion.ipynb se entrenan los modelos con los parámetros econtrado en los ajustes de cada cuaderno asociado a los modelos.

## Evaluar rendimiento
En cada cuaderno asociado a los modelos se imprimen reportes de clasificación con accuracy, precisión, recall, f1;
matriz de confusión, roc-auc curvas. La optimización de los parametros de los modelos se baso en la métrica f1 debido, como ya
se mensiono anteriormente, al desbalance de la muestra a favor de la clase 'No', favoreciendo f1 a las clases postivias , 
minoritaria en nuestro caso.

En el cuaderno ns-4-comparacion.ipynb se muestra un grafico de lineas, asociado a la metricas ROC-AUC, donde se comparan los cinco
modelos. Se puede ver que el clasificador MLP es el que mejor minimiza la deteccion de falsos positivos, pero no es tan bueno para los positivos

## Hacer predicciones
En cada cuaderno asociado a los modelos se muestran gráficas de las predicciones. En el cuaderno ns-4-comparacion.ipynb 

|   Algoritmo       |  Clase | Precision | Recall | f1-score |  support |
| ----------------- | ------ | --------- | ------ | -------- | -------- |
| LogisticRegresion |   0    |    0.8971 | 0.7382 |   0.8099 |      508 |
|	            |   1    |    0.5333 | 0.7795 |   0.6333 |      195 |
| RandomForest      |   0    |    0.8595 | 0.8071 |   0.6081 |      508 |
|                   |   1    |    0.5664 | 0.6564 |   0.6081 |      195 |

se muestran las variables mas relevantes encontradas por RandomForest.

|  No. |      Variable       | Importancia  |
| ---- | ------------------  | ------------ |
|  14  |  Contract           |     0.170747 |
|  17  |  MonthlyCharges     |     0.159277 |
|  18  |  TotalCharges       |     0.154168 |
|   4  |  tenure             |     0.151599 |
|   8  |  OnlineSecurity     |     0.053782 |
|  11  |  TechSupport        |     0.045133 |
|  16  |  PaymentMethod      |     0.044922 |
|   7  |  InternetService    |     0.035698 |
|  15  |  PaperlessBilling   |     0.021422 |
|   9  |  OnlineBackup       |     0.021016 |
|   0  |  gender             |     0.021005 |
|  13  |  StreamingMovies    |     0.019183 |
|   6  |  MultipleLines      |     0.018310 |
|   3  |  Dependents         |     0.017627 |
|   2  |  Partner            |     0.016953 |
|  10  |  DeviceProtection   |     0.016598 |
|  12  |  StreamingTV        |     0.015608 |
|   1  |  SeniorCitizen      |     0.012041 |
|   5  |  PhoneService       |     0.004911 |

## Comparaciones entre modelos
En el cuaderno ns-4-comparacion.ipynb se ejecutan los 5 algorimos vistos anteriormente con los parametros ya ajustados. 
Se calculan y comparan las metricas f1 y roc-auc. El modelo de RandomForest muestra un mejor resultado para la clasificación, 
a partir  de la comparacion de la siguiente tabla:

|       Modelo    |    f1    |  roc_auc |
| --------------- | -------- | -------- |
| RandomForest    | 0.647826 | 0.767878 |
| GradientBoost   | 0.603550 | 0.721184 |
| MLPClassifier   | 0.555911 | 0.692565 |
| SVC             | 0.563981 | 0.698829 |
| RegressionLog   | 0.629423 | 0.763058 |


En el cuaderno tambien se mustran graficos de barra para comparar ambas metricas por separado. Y se plotea un grafico de 
linea la curva roc-auc para cada algoritmo.


## Conclusiones
1- Se trabajo con una muestra desbalanceada a favor de los casos negativos. Muestra que no se pudo remuestrear. Por ello
la comparación se apoyo mas en la métrica f1 y roc-auc.

2- Se le dio mayor peso a las instancias de la clase positiva, clase minoritaria, en los modelos que lo permitieron, 
RandomForest y LogisticRegression. Modelos con mejores resultados.

3- Se trabajo implicitamente con Validación cruzada en el ajuste de los parámetros de los modelo. 

4- Personalmente no estoy satisfecho con la capacidad de discriminación del modelo LogisticRegression, pero teniendo 
encuenta la muestra y las condiciones de la misma es un buen punto de partida para enfrentar el problema propuesto.

5- Entre las variables mas relevantes que ayudan a clasificar a un cliente estan el tipo de contrato, el pago mensual,
el tiempo en la empresa y la seguridad online.

## Recomendaciones

1- Depurar un poco más el conjunto de datos con que se entrenaron los modelos. Identificar los falsos positivos que estan mas próximo del umbral de clasificación, de ser posible realizar encuestas de satisfacción a estos clientes.

2- Profundizar en la teoría, ventajas y desventajas de los modelos utizados, asi como su adaptación al problema planteado.

3- Profundizar en los parámetros tecnicos, de sklearn, en los diferentes modelos utilizados, con vista a reajustar optimamente 
los modelos a través de GridSearchCV.

4- Revisar los modelos que no admitieron ajustar el peso a la clase minoritaria para ver si tienen algun parametro o 
coeficiente parecido.

5- Valorar probar con otros algoritmos para ver cuanto mejora o empeora.

6- Comparar los modelos a partir del modelo entrenado por GridSearchCV que lo hizo con validación cruzada. 
Comparar los resultados con los del cuaderno ns-4-comparacion.ipynb


## Bibliografía

* https://scikit-learn.org/

* https://github.com/shahumar/Free-Machine-Learning-Books/blob/master/book/scikit-learn%20Cookbook%20-%20Second%20Edition.pdf

* http://powerunit-ju.com/wp-content/uploads/2021/04/Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.pdf

* https://datascience.stackexchange.com/questions/33286/how-to-print-a-confusion-matrix-from-random-forests-in-python
