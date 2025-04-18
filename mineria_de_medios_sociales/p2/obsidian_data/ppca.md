- #pca | #probabilistic | #ml | #ppca

# Concepto
- La [[pca|PCA]] probabilística es una variante que introduce un marco estadístico para la reducción de dimensionalidad.
- Se asumen que los datos observados $x\in\mathbb{R}^{D}$ son **generados** a partir de un espacio latente de menor dimensión $z\in\mathbb{R}^{M}$, junto con ruido *gaussiano*, es decir:
$$x=Wz+\mu+\epsilon$$
- Donde $W$ es una matriz de pesos $D\times M$ que mapea el espacio latente al espacio observado. $\mu$ es la media de los datos en el espacio observado y $\epsilon\sim N(0, \sigma^{2}I)$ es el ruido isotrópico *gaussiano*.
- Una de sus ventajas frente a [[pca|PCA]] es que es capaz de manejar datos muy ruidosos, de forma que parte de la información perdida con el ruido, es reinterpretada y añadida. Esto es porque **PPCA** es capaz de estimar la varianza del ruido como parte de su modelo.

# Generación
- Un dato observado $x$ es generado se la siguiente forma:
	- Se extrae un valor $\hat{z}$ de la variable latente de la [[apriori distribution|distribución a priori]] $P(z)$.
	- A continuación se extrae un valor $x$ de una distribución *gaussiana* isotrópica con media $W\hat{z}+\mu$ y covarianza $\sigma^{2}I$.
- Un ruido isotrópico agrega "esfera de incertidumbre" al espacio observado. Si representáramos este ruido gráficamente, se manifestaría como una nube de puntos con forma esférica alrededor de cada punto generado, en lugar de ser alargada o inclinada hacia una dirección particular.
![[ppca.png]]

# Estimación de parámetros por máxima verosimilitud
- Se supone un conjunto de datos $X=\{x_{N}\}$ con $N$ instancias.
![[ppca-dag.png|300]]
- La [[likelihood function|función de verosimilitud]] es:
$$ln(P(X|\mu,W,\sigma^{2}))=\sum_{n=1}^{N}ln(P(x_{n}|W,\mu,\sigma^{2}))=-\frac{N}{2}\{Dln(2\pi)+ln|C|+Tr(C^{-1}S)\}$$
- Donde $S$ es la matriz de [[covariance|covarianza]] muestral. Si se realizase un promedio sobre muestras muestras, se tendría que:
$$E(S)=WW^{T}+\sigma^{2}I$$
- La maximización con respecto a $W$ puede escribirse como:
$$W_{ML}=U_{M}(L_{M}-\sigma^{2}I)^{\frac{1}{2}}R$$
- Donde $U_{M}$ es una matriz de $D\times M$ cuyas columnas son cualquier subconjunto de tamaño $M$ de los autovectores de la matriz de covarianza muestral $S$.
- La matriz diagonal $L_{M}$ de tamaño $M\times M$ contiene los correspondientes autovalores $\lambda_{j}$ de la matriz de covarianza muestral $S$.
- $R$ es una matriz [[ortonormal]] arbitraria de tamaño $M\times M$.
- El correspondiente estimador de la varianza es:
$$\sigma^{2}_{ML}=\frac{1}{D-M}\sum_{i=M+1}^{D}\lambda_{i}$$
- Donde $\lambda$ se refiere a los autovalores, en concreto a los autovalores de mayor tamaño.
- La matriz de covarianza marginal es:
$$C=WW^{T}+\sigma^{2}I$$

# Distribuciones condicionales y marginales
$$p(z|x)=N(z|M^{-1}W^{T}x,\sigma^{2}M^{-1})$$
- Donde $M=W^{T}W+\sigma^{2}I$.
$$p(x|z)=N(x|Wz,\sigma^{2}I)$$