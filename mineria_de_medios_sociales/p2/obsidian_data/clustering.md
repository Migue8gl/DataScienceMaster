- #ml | #datamining | #clustering

# Concepto
 - El *clustering* es una técnica de aprendizaje no supervisado.
 - En este tipo de problema no se tienen etiquetas o *labels* de los datos, por tanto no se conocen los distintos grupos que existen ni se puede evaluar el rendimiento de algoritmos de aprendizaje supervisado, ya que no se pueden obtener métricas derivadas de los errores cometidos (no se conoce si se están realizando errores o no).
 - Las técnicas de *clustering* permiten dividir el espacio de los datos en distintos grupos o *clusters*.
 - Estos *clusters* son conformados por elementos similares entre sí.

# K-Means
- El algoritmo de *K-means* es el algoritmo clásico de *clustering* y uno de los más usados.
## Algoritmo
- **Entrada** -> $k$ *clusters* y $m$ objetos.
- **Procedimiento**:
	- 1 - Se eligen de manera aleatoria $k$ centroides (centros de las agrupaciones).
	- 2 - Repetir mientras haya cambios o no muchos cambios (un umbral):
		- 2.1 - Asignar cada objeto al *cluster* con centroide más cercano.
		- 2.2 - Recalcular los centroides como el punto medio de la agrupación creada.
## Características
- Se intenta minimizar la suma de las distancias entre cada objeto con respecto al centroide del *cluster* al que pertenece.
- Una de las **desventajas** del algoritmo es que dependiendo de la inicialización, los *clusters* pueden variar mucho. Tienden a estancarse en **mínimos locales**. Una solución para ello es realizar *K-means* múltiples veces y escoger aquella solución que minimice la [[objetive function|función objetivo]].
- **Ventajas**:
	- Muy simple y eficaz para *clusters* esféricos.
	- Relativamente rápido $O(k\cdot m\cdot iter)$.
- **Desventajas**:
	- Es muy sensible a [[outliers]], ya que estos afectan a la [[mean and weighted mean|media]].
	- No es capaz de detectar patrones no esféricos.
	- Dificultad de manejo de datos nominales.
	- Es difícil determinar el valor de $k$.

# Distancias
- Un componente clave del *clustering* es el tipo de distancia que se utilice, pues dependiendo de esta se podrán medir distintos tipos de similitud entre individuos.
- [[distances]]

# Métricas
- Existen métricas de calidad para *clustering*. 
- El problema es que estas medidas no son tan objetivas como en el aprendizaje supervisado y por tanto es necesario tener conocimiento sobre los datos y lo que se pretenda extraer de estos *clusters*.
- **Métodos Intrínsecos**: Estos métodos evalúan la calidad de las agrupaciones utilizando únicamente las propiedades internas de los datos y sus agrupaciones sin comparar con ningún resultado externo o de referencia. Los criterios de evaluación incluyen:
	- **Número de elementos**: Cuántos datos hay en cada grupo.
	- **Densidad de elementos**: Qué tan concentrados están los datos dentro de cada grupo.
	- **Distancia entre las agrupaciones**: Qué tan separadas están unas agrupaciones de otras.
- **Métodos Extrínsecos**: Estos métodos, por el contrario, requieren un **ground truth**, es decir, una referencia externa de cómo deberían ser las agrupaciones ideales. Aquí, la evaluación se basa en la comparación de las agrupaciones obtenidas con estas agrupaciones ideales conocidas. Este método es adecuado cuando ya existe información previa sobre la estructura de los datos y se desea validar si las agrupaciones encontradas coinciden con la expectativa.
- Algunas medidas intrínsecas son:
	- [[davies-bounding index]]
	- [[silhouette]]
	- [[dunn-index]]
- Para medir cual es el "mejor" $k$ -> [[elbow method]]

# Alternativa
- [[k-medoids]]