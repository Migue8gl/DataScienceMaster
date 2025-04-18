- #ml | #statistics | #pca

# Concepto
- Es un método de **reducción de dimensionalidad** que transforma las columnas de un conjunto en un nuevo conjunto de características llamadas componentes principales.
- La idea es reducir variables reteniendo la máxima cantidad de información posible.
- La información de una columna se mide en la cantidad de variación que contiene.

# Requisitos
- Debe existir correlación lineal entre las características.
- No deben existir [[outliers]].
- Variables continuas.
- Los datos deben seguir una [[normal distribution|distribución normal]] (aprox).
- Fechas a intervalos.

# Matemáticas
- Sea $$X^{T}=[x_{1},x_{2},...,x_{I}]_{D\times I}$$
- $I$ es el número de vectores de rasgos, $D$ es el número de componentes que tiene cada vector de rasgos.
- El objetivo de la reducción de dimensionaldad (de los rasgos) es encontrar una representación de menor dimensión $h$ que pueda explicar el dato de manera aproximada.
$$x\approx f(h,\theta)$$
- Donde $f$ es una función de variables ocultas y un conjunto de parámetros $\theta$. Normalmente se elige una familia de funciones y entonces se aprenden $f$ y $\theta$ de un conjunto de entrenamiento.
## Criterio de mínimos cuadrados
$$\hat{\theta},\hat{h}_{1...I}=argmin_{\theta,h_{1...I}}\left[\sum_{i=1}^{I}(x_{i}-f(h_{i},\theta))^{T}(x_{i}-f(h_{i},\theta))\right]$$
- Elegir los parámetros $\theta$ y las variables ocultas $h$ de forma que minimicen el [[mean squared error|error cuadrático medio]] de aproximación.

- La transformación para aproximar $x$ es:
$$x_{i}\approx\theta h_{i}+\mu$$
- Donde $x_{i},\mu,\theta$ son vectores columnas y $h_{i}$ es un escalar.
- El vector $\theta$ es un conjunto de parámetros, que define las direcciones principales en el espacio de datos. Cada columna es una dirección en el espacio de características.
- El escalar $h_i$ es un coeficiente que indica cuanto contribuye la matriz de direcciones para aproximar $x_{i}$.
- Si en el preprocesamiento se le resta la media a los datos, obtenemos datos centrados (con media igual a cero) y ya no haría falta sumar $\mu$.
- Al solucionar el problema de minimización, encontramos que la solución no es única (hay infinitas). Si se multiplicamos $\theta$ por una constante y dividimos cada variable $h_i$ por la misma constante, se obtiene el **mismo coste**. Por ello se impone que la norma (magnitud de la matriz) de $\theta$ sea $1$ usando multiplicadores de *Lagrange* (es una forma de minimizar con restricciones).
## Nueva función de coste
$$E=\sum_{i=1}^{I}(x_{i}-\theta h_{i})^{T}(x_{i}-\theta h_{i})+\lambda(\theta^{T}\theta - 1)$$
- Donde se añade el término de *Lagrange*:
	- $\theta^{T}\theta$ es la norma al cuadrado de $\theta$. Se le resta $1$ porque se requiere que sea igual a $1$. Al hacer esto, cuando la norma sea $1$ este término no penalizará. Se intentará hacer el término lo má cercano a cero posible.
	- $\lambda$ es el multiplicador de *Lagrange*, que penaliza cuando esa restricción no se cumple.
- Se minimizarla se deriva con respecto al valor escalar y a la matriz de parámetros, se iguala a cero y se reordena.
- Derivando con respecto a $h_{i}$ se obtiene:
	$$\hat{h}_{i}=\hat{\theta}^{T}x_{i}$$
- Solo se debe computar el [[dot product|producto escalar]] del vector inicial de rasgos con el vector $\theta$.
- Derivando con respecto a $\theta$ se obtiene:
$$\begin{align}\sum_{i=1}^{I}x_{i}x_{i}^{T}\hat{\theta}=\lambda\hat{\theta} \\ X^{T}X\hat{\theta}=\lambda\hat{\theta}\end{align}$$
- Para calcular el vector $\theta$, calculamso el primer [[eigenvectors and eigenvalues|autovector]] (el que tiene un $\lambda$ mayor) de la matriz de dispersión $X^TX$, lo cual significa escoger la dirección con mayor varianza.

# Pasos simplificado
- $1$. Se obtiene la matriz de [[covariance|covarianza]].
- $2.$ Se descompone la matriz en sus autovectores y autovalores.
- $3.$ Se ordenan descendientemente por el valor de los autovalores y se obtienen los $k$ autovectores.
- $4.$ Se reconstruye la matriz de transformación.

# Problema en altas dimensionalidades
- Si $X$ es una matriz $I\times D$, entonces $X^{T}$ es $D\times I$. Por lo que $X^{T}X$, o lo que es lo mismo, la matriz de dispersión (o covarianza si los datos están centrados y normalizados) tiene tamaño $D\times D$ (es inmensa).
- Si se pudiese trabajar con $XX^{T}$ la matriz sería $I\times I$.
- Para ello se trabaja con la descomposición $SVD$.