- #variance  | #standarddeviation | #desviaciónestandar

### Varianza y desviación estándar en la población
- Es un **estadístico de dispersión.**
- La *varianza* es una medida que nos aporta información de como de repartidos están nuestros datos, es decir, la diferencia de distancia entre nuestros datos y la media.
- Básicamente, la varianza es una media de la distancia entre cada valor y la [[mean and weighted mean|media]] de nuestro conjunto de datos. Se suman todos los valores, restando el valor de cada $x_i$ con el de la media. Luego, esa suma la elevamos al cuadrado. Esto se hace para que los negativos no cancelen a los positivos y para agrandar valores cuya diferencia inicial sea más grande, de forma que se "pondera" más aquellos valores más alejados.
- La fórmula es la que sigue:
	- $\sigma^2=\frac{\sum{(x_i-\mu)^2}}{N}$
- Como podemos observar, la varianza está dada al cuadrado, si tomamos la raíz cuadrada de la varianza obtenemos la **desviación estándar**. Es la varianza expresada en la escala de nuestros datos.

### Varianza y desviación estándar en la muestra
- $s²=\frac{\sum{(x_i-\overline{x})^2}}{n-1}$
- $95\%\rightarrow$ media $\pm 2\cdot s$
- $70\%\rightarrow$ media $\pm s$
- La diferencia con respecto a la población, es que contamos uno menos a la hora de calcular la varianza. Esto se hace para incrementar el valor de esta, de forma que se incrementa la incertidumbre en la muestra.
- Esto es interesante porque no queremos subestimar la varianza de la muestra, queremos hacer decrecer cualquier [[bias|sesgo]] que haya podido capturar nuestro conjunto de datos.
- Las muestras son **siempre** conjuntos de datos que, por muy bien que se hayan obtenido, van a contener ruido y sesgos.
- Básicamente, reducimos la confianza de que los datos estén bien ajustados.