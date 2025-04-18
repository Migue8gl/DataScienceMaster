- #closedformequation | #linearregression | #inversematrix

# Ecuación de fórmula cerrada
- Fórmula para ajustar de manera exacta una regresión lineal de una sola variable de entrada, de forma que no sirve para la mayoría de problemas de regresión lineal.
	- $m=\frac{n\sum xy-\sum x\sum y}{n\sum x² -(\sum x)^2}$
	- $b=\frac{\sum y}{n}-m\frac{\sum x}{n}$

# Pseudoinversa
- Si $X^T · X$ es invertible, entonces $X^{\dagger}=(X^T · X)^{-1}·X^T$ es la pseudoinversa de $X$.
	- $m=X^{\dagger}·y$, donde $y$ son los outputs o etiquetas.
	- $b=mean(y)-m·mean(X)$ (aproximado)

![[example-linear-regression-pseudoinverse-and-sklearn.png]]

- Podemos calcular la **pseudoinversa** a través de la descomposición de la matriz de inputs (descomposión SVD). La fórmula es la que sigue:
	- $X^{\dagger}=V·D^{\dagger}·U^T$
	- Donde $V$ es la matriz singular derecha de la descomposión de $X$, donde $U^T$ es la matriz singular derecha transpuesta, y donde $D^{\dagger}$ es la matriz pseudoinversa diagonal de $X$.
- Si no fuese invertible, la pseudoinversa no serı́a definible, pero en la mayorı́a de casos esto es posible, por lo que no habrı́a problema. De esta forma, obtenemos el [[vector]] de pesos de un solo paso, esta es una de las grandes ventajas de usar regresión lineal.
- También existen otros tipos de descomposión, como [[eigenvectors and eigenvalues#^049f57|descomposición por valores propios]], o [[qr decomposition|descomposición QR]].

# Gradiente descendiente
- [[gradient descent]].