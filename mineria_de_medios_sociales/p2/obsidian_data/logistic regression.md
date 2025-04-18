- #ml | #logistic | #probability | #regression | #classification
- links: [Variable de Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) -> Función de probabilidad a partir de interpretación de las clases como variables de Bernoulli.

## Introducción
- La regresión logística es un algoritmo de clasificación usado en el ámbito del aprendizaje automático. Originalmente, este algoritmo es un clasificador binario, aunque existen formas de generalizarlo a una extensión multiclase.

## Reglas
- Bajo el modelo de regresión logística se asumen relaciones lineales entre los pesos ponderados $w$ y los [[logits]] (probabilidades a favor de un o unos eventos concretos).
	- $logit(p)=\sum_{i=1}^Nw_ix_i+b=w^Tx+b$
	- Se describe una relación entre los *odds* logarítmicos y [[perceptron#^de0f30|net input]].
- $p$ es pues la probabilidad de que clasificación en una clase concreta, una [[conditional probability|probabilidad condicional]] donde una ejemplo es de la clase $y$ dados unas características $x$, o lo que es lo mismo $p(y|x)$.
- La inversa de la función logit es la [[transfer functions#^bda699|sigmoidal]] o función logística sigmoidal:
	- $\sigma(z)=\frac{1}{1+e^{-z}}$
	- Donde $z$ es el *net input*.
- Por tanto, si $z$ tiende al infinito, la función sigmoidal se acerca a $1$. De forma contraria, cuando el *net input* se acerca al menos infinito, la sigmoidal a $0$. Gracias a este transferencia de valores reales al rango $[0,1]$, estos se pueden interpretar como probabilidades.

![[logistic regression-diagram.png]]

- Las probabilidades se pueden convertir a un valor concreto por medio de una función umbral.

## Actualización de pesos
- La función de pérdida logística se define de la siguiente manera:
	- La [[likelihood function|función de probabilidad]] que se intenta maximizar, asumiento que los ejemplos en el conjunto de datos son independientes unos de otros, es la siguiente:
		- $L(w,b|x)=p(y|x;w,b)=\prod_{i=1}^Np(y_i|x_i;w,b)=\prod_{i=1}^N(\sigma(z_i))^{y_i}(1-\sigma(z_i))^{1-y_i}$
		- La expresión  $(\sigma(z_i))^{y_i} (1 - \sigma(z_i))^{1 - y_i}$ puede interpretarse así:
			- Si $y_i = 1$: 
				- $p(y_i | x_i; w, b) = \sigma(z_i)$ 
				- Esto representa la probabilidad de que la i-ésima observación sea clasificada como $1$. 
			- Si $y_i = 0$: 
				- $p(y_i | x_i; w, b) = 1 - \sigma(z_i)$ 
				- Esto representa la probabilidad de que la i-ésima observación sea clasificada como $0$. La combinación de estas probabilidades para todas las observaciones da la probabilidad conjunta: 
				- $\prod_{i=1}^N (\sigma(z_i))^{y_i} (1 - \sigma(z_i))^{1 - y_i}$
	-  En la práctica, es más fácil maximizar el logaritmo natural de esta ecuación (**log likelihood**). 
		- $l(w,b|x)=log(L(w,b|x))=\sum_{i=1}^N[y_ilog(\sigma(z_i))+(1-y_i)(log(\sigma(z_i)))]$
		- Los logaritmos ayudan a evitar el [[numerical stability|underflow]] para probabilidades muy bajas. Además, se puede convertir el producto de factores en una sumatoria, lo que facilita la derivada -> [[logarithms#^fa3850|Propiedades de los logaritmos]].
	- Esta función ha de ser maximizada, pero puede reescribirse para minimizarse y así obtener una función de pérdida.