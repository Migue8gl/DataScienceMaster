- #linealgaussianmodel | #bayesiannetwork 

# Concepto
- Se puede modelar la [[decomposition theorem|CPD]] $P(Y|X)$ como una gaussiana, cuyos parámetros dependen del valor de $X$.
- Una solución muy común es que la media de $Y$ sea una función lineal de $X$ y que la varianza de $Y$ no dependa de $X$. Por ejemplo:
	- $P(Y|x)=\mathcal{N}(-2x+0.9;1)$
- Sea $Y$ una variable continua con padres continuos $X_{1},...,X_{k}$. Decimos que $Y$ sigue un modelo lineal gaussiano si hay parámetros $\beta_{0},...,\beta_{k}$ y $\sigma^{2}$ tal que:
	- $P(Y|x_1, \dots, x_k) = \mathcal{N}(\beta_0 + \beta_1 x_1 + \dots + \beta_k x_k, \sigma^2)$
- O sea, $Y$ es una función lineal de las variables padre, con la suma de un ruido gaussiano $\epsilon$ de media $0$ y varianza $\sigma^2$:
	- $Y=\beta_{0}+\beta_{1}x_{1}+...+\beta_{k}x_{k}+\epsilon$
- [[gaussian bayesian network]]