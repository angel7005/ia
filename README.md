# Hackathon de IA de NataSquad
## Problema de machine learning
### Problema 1 Customer Churn.

https://hackathon.natasquad.com/

https://github.com/Natasquad-Hack/NataSquad-AI-Hackathon

## Introducción

Para una familiarizacion con el problema leer el documento Customer_Churn.md, lo mismo en la carpeta actual que en el projecto 
https://github.com/Natasquad-Hack/NataSquad-AI-Hackathon/blob/main/1_Machine_Learning/1_Customer_Churn/Customer_Churn.md

En las mismas referencias puede ser encontrado el archivo archive.zip con el juego de datos. En el se puede observar el cliente, las variables predictoras y la variable objetico(Churn). Un poco mas de 7000 instancias de clientes.

Entre las variables predictoras hay variables de tipo boolean, categoricas, enteras y continuas. Menos de 20 variables.

La variable objetivo (churn) tiene dos valores validos: No, Yes. Nos interesa predecir a los clientes con valor 'Yes', que abandonaran la empresa.
De la observacion de los datos proporcionados se observa una muestra sesgada o con desbalance a favor de la clase negativa 'No', con mas del 75%. 

Esto puede dificultar el aprendisaje del modelo al contar con pocos clientes con  valor 'Yes' en la variable objetivo, en comparacion 
con el valor 'No' que es la clase mayoritaria

La seleccion de variables con ayuda del especialista puede ayudar a depurar la muestra de clientes suministrada,
comensando por los falsos positivos y próximo a la frontera o al umbral de clasificación según el modelo

Encontrar las variables mas relevantes en la clasificación de los clientes, ayudara objetivamente a la empresa a saber donde puede invertir su dinero para retener al cliente. 

La solucion pasa por tener en cuenta el problema de desbalance de la muestra, predecir si un cliente abandonara la empresa y cuales son 
las variables mas importantes que influyen en que el cliente tome esa decision.


## Solucion

Dado que el problema se perfila como un problema de clasificación binaria con aprendizaje, los algoritmos candidatos
son RandomForest, GradientBoosting, MLPClassifier, SVC, LogisticRegression. 

Para lo que se crea un cuaderno por cada modelo y uno para la comparación.

Preliminarmente se hicieron algunas corridas con los datos de pruebas proporcionados y se compararon las métricas 
resultados de diferentes algoritmos (RandomForest, GradientBoosting, MLPClassifier, LogisticRegression, SVM). Hice una busqueda en 
google para la recomendación de algoritmos alternativo al RandomForest y estos fueron los ganadores. El proximo paso seria
actualizarme en unos y conocer en otros de las ventajas y desventajas de los algoritmos alternativos.

Retomando el problema del desbalance de la muestra se valora entre las técnicas propuestas el remuestreo y ajuste de los parámetros del algoritmo dando mas peso a la clase minoritaria ('Yes').

El proyecto se desarralla con python como lenguaje de programación y las bibliotecas sklearn de scikit-learn. Principalmente con el 
algoritmo RandomForest y la clase GridSearchCV que es una técnica de ajuste de hiperplanos para econtrar la combinación óptima de 
parametros del algoritmo y evitar el OverFitting o sobre ajuste utilizando la validación cruzada.


## Preparacion y depuracion de los datos
Al cargarse los datos se encuentran valores omitidos en la penúltima columna "TotalCharge". 


Cargos totales vacios, los pongo a 0.0
	 	los visualice y coincide con que:
	 	a) La variable tenure es "0",  el numero de meses(tenure) que el cliente
 		  ha estado en la empresa es cero.
 		b) La variable Churn tienen valor "No", Por lo que se deduce
 		que no ha tenido tiempo de cobrar la mensualidad o no han podido pagar aun.
 		c) Dado que la muestra esta sesgada a la clase 'No', pudiera simplemente no incluirla en el entrenamiento e incluirla en la 
 		submuestra de prueba. Pero asocie a estos clientes como clientes recien incorporados al sistema.
    
Se observan muchas variables categoricas, con las que los modelos no pueden trabajar.
       Se tranforman en valores numericas.

## Datos de entrenamiento y pruebas
Los datos de separaron 90% para entrenamiento y el ultimo 10% para prueba. Ahora el 90% de entrenamiento se le dio a GridSearchCV, se probó con 5, 7 y 9 particiones sobre el que se ajustaron los parametros del modelo

## Seleccionar modelo y ajustar parametros
Se crea un cuaderno por cada modelo donde se van a optimizar los parámetros de cada modelo. Utilizando GridSearchCV

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
En el cuaderno  ns-4-comparacion.ipynb se entrenan los modelos con los parametros econtrado en los ajustes de cada cuaderno asociado a los modelos.

## Evaluar rendimiento
En cada cuaderno asociado a los modelos se imprimen reportes de clasificacion con accuracy, precision, recall, f1;
matriz de confusion, roc-auc curvas. La optimizacion de los parametros de los modelos se baso en la metrica f1 debido, como ya
se mensiono anteriormente, al desbalance de la muestra a favor de la clase 'No'

## Hacer predicciones
En cada cuaderno asociado a los modelos se muestran graficas de las predicciones. En el cuaderno ns-4-comparacion.ipynb 
se muestran las variables mas relevantes encontradas por RandomForest.
           variable  importancia
14          Contract     0.170747
17    MonthlyCharges     0.159277
18      TotalCharges     0.154168
4             tenure     0.151599
8     OnlineSecurity     0.053782
11       TechSupport     0.045133
16     PaymentMethod     0.044922
7    InternetService     0.035698
15  PaperlessBilling     0.021422
9       OnlineBackup     0.021016
0             gender     0.021005
13   StreamingMovies     0.019183
6      MultipleLines     0.018310
3         Dependents     0.017627
2            Partner     0.016953
10  DeviceProtection     0.016598
12       StreamingTV     0.015608
1      SeniorCitizen     0.012041
5       PhoneService     0.004911

## Comparaciones entre modelos
En el cuaderno ns-4-comparacion.ipynb se ejecutan los 5 algorimos vistos anteriormente con los parametros ya ajustados.
Se calculan y comparan las metricas f1 y roc-auc. El modelo de LogisticRegression es el que mejor resultado muestra 
para la clasificacion

## Conclusiones
1- Se trabajo con una muestra desbalanceada a favor de los casos negativos. Muestra que no se pudo remuestrear. Por ello
la comparacion se apoyo mas en la metrica f1 y roc-auc.

2- Se le dio mayor peso a las instancias de la clase positiva, clase minoritaria, en los modelos que lo permitieron, 
RandomForest y LogisticRegression. Modelos con mejores resultados.

3- Se trabajo implicitamente con Validacion cruzada en el ajuste de los parametros de los modelo. 

4- Personalmente no estoy satisfecho con la capacidad de discriminacion del modelo LogisticRegression, pero teniendo 
encuenta la muestra y las condiciones de la misma es un buen punto de partida para enfrentar el problema propuesto.

5- Entre las variables mas relevantes que ayudan a clasificar a un cliente estan el tipo de contrato, el cargo mensual,
el tiempo en la empresa y la seguridad online.

## Recomendaciones

1- Depurar un poco mas el conjunto de datos con que se entrenaron los modelos. Identificar los falsos positivos que 
estan mas proximo del umbra de clasificacion, de ser posible realizar encuestas de satisfaccion a esos clientes.

2- Profundizar en la teoria, ventajas y desventajas de los modelos utizados, asi como su adaptacion al problema planteado.

3- Profundizar en los parametros tecnicos, de sklearn, en los diferentes modelos utilizados, con vista a reajustar optimamente 
los modelos a travez de GridSearchCV.

4- Revisar los modelos que no admitieron ajustar el peso a la clase minoritaria para ver si tienen algun parametro o 
coeficiente parecido.

5- Valorar probar con otros algoritmos para ver cuanto mejora o empeora.

6- Comparar los modelos a partir del modelo entrenado por GridSearchCV que lo hizo con validacion cruzada. 
Comparar los resultados con los del cuaderno ns-4-comparacion.ipynb


## Bibliografia

https://scikit-learn.org/

https://github.com/shahumar/Free-Machine-Learning-Books/blob/master/book/scikit-learn%20Cookbook%20-%20Second%20Edition.pdf

http://powerunit-ju.com/wp-content/uploads/2021/04/Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.pdf

https://datascience.stackexchange.com/questions/33286/how-to-print-a-confusion-matrix-from-random-forests-in-python
