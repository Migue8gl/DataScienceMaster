- #logisticfunction | #sigmoid

- La función logística no es otra que la función señal o **sigmoidal**. Este tipo de función produce valores entre $0$ y $1$ de forma que los valores pueden ser interpretados como una probabilidad.
- La función logística es la siguiente:
	- $y=\frac{1}{1+e^{-(\beta_0+\beta_1x)}}$
- Esta fórmula utiliza el [[euler's number|número de euler]]. La $x$ es la variable independiente o variable de entrada. $\beta_0$ y $\beta_1$ son los coeficientes para los que necesitamos resolver.
- De hecho $\beta_0$ y $\beta_1$ son equivalentes a la **pendiente** e **intersección** de la [[essential math for data science#Linear Regression|regresión lineal]].
- $\beta_0 = b$ y $\beta_1=m$.

### Regresión logística multivariable
- $y=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_nx_n)}}$