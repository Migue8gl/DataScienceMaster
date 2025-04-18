- #missingvalues 

# Tipos
- Se pueden utilizar las siguientes opciones, aunque algunas de ellas sesgan los datos:
	- Ignorar la tupla. Suele usarse cuando la variable a clasificar no tiene valor.
	- Rellenar manualmente los datos. En general es impracticable.
	- Utilizar una constante global para la sustitución.
	- Rellenar utilizando la media/desviación del resto de las tuplas.
	- Rellenar utilizando la media/desviación del resto de las tuplas pertenecientes a la misma clase.
	- Rellenar con el valor más probable. Para ello utilizar alguna técnica de inferencia, p.e. bayesiana o un [[decision trees|árbol de decisión]].
## Media/Desviación típica
- La imputación por [[mean and weighted mean|media]] y [[variance and standard deviation|desviación típica]] se realiza generando un valor aleatorio alrededor de la media usando esa desviación.
```python
def random_impute(): return np.random.normal(mean_value, std_dev)
```
- Cuando la distribución no es [[normal distribution|normal]] no suele ser una buena idea, ya que añade [[bias|sesgo]] a los datos.
- En caso de poder usarse es una buena opción pues añade variabilidad a los datos faltantes.
## Moda
- La [[mode|moda]] suele usarse para variables categóricas. Pero si la distribución de estos valores es parecida, tienen probabilidad similar, no es un buen método.
## KNN
- Se escogen los $k$ vecinos más cercanos del elemento con valor faltante y este se imputa por haciendo la media de los valores de los vecinos para ese atributo.
## Weighted KNN
- Similar al método de **KNN** pero utilizando ponderaciones en los vecinos, de forma que vecinos más cercanos dentro de los $k$ más cercanos tengan más peso.
## Cluster
- Este método primero *clusteriza* los datos y después usa la media o mediana del clúster correspondiente a la observación con el *missing value* para imputar el valor.
## Otros algoritmos de machine learning
- Se pueden usar otros algoritmos como [[support vector machine|SVMs]], [[linear regression|Regresión Lineal]] u otros para entrenarse sobre las características de las observaciones no faltantes, utilizando como *target* la característica faltante, y así predecir el valor de la observación con valor faltante.
## MICE
- El algoritmo *multiple imputation by chained equations* es un método robusto para imputación. 
- El procedimiento imputa los datos que faltan en un conjunto de datos mediante una serie iterativa de modelos predictivos.  
- En cada iteración, cada variable en el conjunto de datos se imputa utilizando las demás variables del conjunto de datos. Estas iteraciones deben ejecutarse hasta que parezca que converge.
![[imputations-mice-alg.png]]
- Normalmente no requiere de demasiadas iteraciones (entre $5-10$.
- Tiene en cuenta las relaciones entre variables.