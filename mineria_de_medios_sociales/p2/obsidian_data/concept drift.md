- #datascience | #ml | #drift 

# Concepto
- El *concept drift* se refiere a una dificultad que surge del aprendizaje de datos según estos fluyen con el tiempo. El cambio de concepto de los datos implica que un modelo aprendido en el pasado ya no es consistente con los datos recibidos en el presente.
- Un tratamiento adecuado con este efecto involucraría la actualización o creación de un nuevo modelo que tenga en cuenta nuevas características que entren en conflicto con asunciones pasadas, pero conservando asunciones basadas en datos antiguos que sigan cumpliendose.
- El *concept drift* es **real** si altera la frontero de decisión y es **virtual** si solo varía la distribución de los datos sin alterar esa frontera.
![[concept drift-virtual-vs-real.png|400]]
![[concept drift-adaptation.png]]

# CD mediante ventana.
- Son algoritmos que incorporan mecanismos para olvidar datos antiguos. Estos se basan en la premisa de que los datos más recientes son más relevantes, por tanto, mediante una ventana deslizante, proceden a crear nuevos conjuntos de datos que entrenan un nuevo modelo o que realizan periódicamente [[online machine learning|aprendizaje incremental]].
# CD mediante Ensemble
- Los algoritmos de [[ensembles]] pueden incrementar la precisión dado que el conocimiento distribuido entre los clasificadores puede ser más exhaustivo.
- Entre los enfoques más destacables están:
	- *Streaming Ensemble Algorithm*.
	- *Accuracy Weighted Ensemble*.
- Ambos algoritmos tienen un número fijo de clasificadores. Los datos que van llegando se recogen en fragmentos para entrenar nuevos clasificadores. Los clasificadores nuevos se evalúan y el peor del *ensemble* original es reemplazado por el mejor de los nuevos si sus métricas son mejores. **SEA** usa voto por mayoría y **AWE** usa voto ponderado.
- Un algoritmo más complejo es **DWM** (*Dynamic Weighted Majority*), que pondera los clasificadores en función de sus decisiones incorrectas. Su número de clasificadores es variable y cada uno de ellos tiene asociado un peso (errores), si ese peso supera un umbral ese clasificador es eliminado.
- Se pueden añadir clasificadores cuando el conjunto toma una decisión errónea.
- También existen algoritmos que usan una bolsa de clasificadores que se actualiza cuando se detecta **CD**, pero no se eliminan los modelos existentes, por lo que los datos pasados se pueden conservar.

# Detección
## DDM (Drift Detection Method)
- Asumen la posibilidad de conocer el rendimiento del modelo que se está aprendiendo. Detectan el desvío de concepto en función de la precisión del clasificador o mediante un análisis de ditribución de clase. Pueden ser algoritmos ineficientes y, por tanto, incapaces de asumir una tasa de llegada de datos elevada.
- *Probability of missclassifying* $(p_{i})\rightarrow s_{i}=\sqrt{\frac{p_{i}(1-p_{i})}{i}}$.
- *Warning Level*: Se empiezan a recoger ejemplos cuando $p_{i}+s_{i}\ge p_{min}+2\cdot s_{min}$.
- *Drift Level*: Se construye un modelo nuevo a partir de los nuevos datos almacenados cuando $p_{i}+s_{i}\ge p_{min}+3\cdot s_{min}$. Se reinician $s_{min}$ y $p_{min}$.
## ADWIN (ADaptative sliding WINdow)
- Dada la ventana $W$, si existen dos subventanas $W_{0}$ y $W_{1}$ suficientemente grandes y con medias suficientemente distintas, se concluye que los valores esperados son diferentes y se elimina la parte antigua de $W$.
## HSP (Histogram-Based Straightforward prediction)
- Se basa en la detección de **CD** utilizando únicamente las muestras y que es muy eficiente.
- Se detecta **CD** si la frontera de decisión cambia. Para eso se utilizan histogramas univariantes para cada característica.
- Estos histogramas se actualizan dinámicamente con cada nueva muestra de datos mediante una función de actualización exponencial y clasifica los datos en función de la distribución en los histogramas.
- Después evalúa cambios en la distribución usando una métrica de rendimiento $G^{*}$. Si $G^{*}$ cambia mucho en el tiempo, indica un **CD**.

# Clustering
- **CluStream** es un algoritmo que divide el proceso en:
	- *Online*: mantiene modelos resumidos de los datos basados en estadísticos (*microclusters*). Estos *microclusters* tienen información sobre la cantidad de datos en el grupo, suma de los valores, suma de los valores al cuadrado, información sobre el tiempo (suma de los tiempos y suma al cuadrado).
	- Cuando llega un nuevo dado se agrega a un *microcluster* si este dato está cerca del mismo. Si no encaja con ninguno se crea un nuevo *cluster* y si existen demasidas agrupaciones se eliminan los *microclusters* más antiguos o se fusionan los más parecidos.
	- *Offline*: usa los *microclusters* previamente almacenados para reconstruir agrupaciones a mayor escala mediante algoritmos como [[k-nearest neighbors|k-means]].
- Los *microclusters* son guardados en instantáneas de tiempo que siguen un patrón piramidal que equilibra la eficiencia de almacenamiento y la capacidad de análisis retrospectivo. Este tipo de ventana de tiempo piramidal que sigue permite analizar los datos en diferentes escalas temporales.
- Al mantener *microclusters* actualizados, **CluStream** puede reflejar cambios en la estructura de los datos sin necesidad de detección explícita de deriva.
- Sin embargo, **no detecta activamente el concept drift**, sino que se adapta implícitamente al actualizar los micro-clusters con nuevos datos
- **StreamDD** es un algoritmo con detección de *drift* explícito que trata de mejorar a **CluStream**. Al igual que este opera en dos fases, una *online* y otra *offline*, en la que se crean *microclusters* y *macroclusters*.
- La diferencia proviene principalmente en la detección de *drift* mediante el *test* de *Page-Hinkley*. Se calcula la distancia máxima promedio entrte *microclusters*, esta distancia es monitorizada y si la suma acumulada de cambios supera un umbral, se detecta *drift*. Esto causa que el modelo se actualice en la fase *offline*.

# Patrones frecuentes
- Se refieren los patrones frecuentes a [[association rules|items frecuentes]] de reglas de asociación.
- Un algoritmo muy potente es **Fuzzy-CSar**, el cual procesa flujos de datos incrementalmente, actualizando reglas en tiempo real. Combina algoritmos genéticos ([[genetic algorithm|AG]]) y mecanismos de crédito para evolucionar reglas.
- Utiliza reglas difusas, que manejan la imprecisión de la información. El sistema procesa datos en tiempo real: cuando llega un nuevo ejemplo, busca reglas existentes cuyas condiciones coincidan parcialmente con los datos. Si no hay suficientes reglas relevantes, genera nuevas reglas mediante un "operador de cobertura", que combina aleatoriamente variables y términos lingüísticos para explorar asociaciones potenciales. Las reglas compiten en "nichos" agrupados por su variable de salida, priorizando aquellas con mayor confianza (fiabilidad de la relación) y soporte (frecuencia de aparición). Para mejorar continuamente, emplea un algoritmo genético que cruza y muta las mejores reglas, añadiendo o eliminando variables en sus condiciones, ajustando términos lingüísticos, o cambiando la variable de salida, lo que permite explorar nuevas hipótesis.