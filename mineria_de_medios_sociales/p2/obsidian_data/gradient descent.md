- #stocasticgradientdescent  | #gradientedescendenteestocástico

### Gradiente Descendiente
- El gradiente descendiente es un algoritmo de optimización de funciones  iterativo. Su función es encontrar un mínimo local de una [[differentiable function|función diferenciable]].
- La idea detrás del algoritmo consiste en avanzar en dirección contraria al [[gradient|gradiente]], de esta forma se avanza de manera que se minimiza el valor de la función objetivo.
- Un algoritmo con el que se puede sacar mucho partido del gradiente descendente es la [[logistic regression|regresión logística]]. Debido a que la función asociada al error de este algoritmo de aprendizaje es una función convexa, solo tiene un mínimo, por lo que si se alcanza mínimo se asegura el ser global.

### Fórmula
- Para dar la fórmula, debemos definir el **learning rate**. Para esta variable utilizaremos el símbolo $\eta$.
- Se actualiza el punto $x_i$ siguiendo el siguiente criterio:
	- $x_{i+1}=x_i-\eta\frac{\partial f(x)}{\partial x_i}=x_i-\eta \triangledown f(x)$
	- Donde $x_i$ es el punto para el que se calcula el gradiente.

### Gradiente descendiente estocástico
- Esta variante surge debido a que, computacionalmente, es muy costoso calcular el gradiente para cada punto de la función, sobre todo si en un ámbito de aprendizaje automático utilizamos un set de entrenamiento muy grande. 
- Para economizar cada iteración del algoritmo, la variación estocástica del gradiente descendiente escoge al azar una [[sample|muestra]] del conjunto completo para cada iteración.
- Pongamos que la función a minimizar es $f(x)$, de forma que $f(x)=\sum{f_i(x)}_{i=1}^n$, donde cada $i$ indica el índice de cada subconjunto. 
- Algunos de sus beneficios son, además del computacional, que reduce el [[overfitting and variance|sobreajuste]], debido a que en cada iteración del algoritmo cambian los batches (en este caso mini-batches) de forma que no se estanca en el mínimo local general, ya que el "escenario" cambia en cada iteración.

### Método iterativo
- Inicializamos los pesos $w=0$ y $\eta=\eta_0$ (el learning rate puede ser adaptativo).
- Iteramos.
	- Mezclamos aleatoriamente la muestra $i$ y la dividimos en mini-batches.
	- Iteramos cada mini-batch.
		- Desde $j=0$ hasta $k$.
			- $w_j=w_j-\eta\triangledown f_i(w_j)$ (solo un minibatch por adaptación)
- Hasta que el mínimo definido se alcance.

### Estandarización de los puntos en el espacio
- El gradiente descendiente de puede beneficiar de una transformación de los datos en el espacio, como es la estandarización. Se encarga de que cada característica o variable tenga una media de cero y una desviación estándar de uno.
	- $x_i^{\prime}=\frac{x_i-\mu_i}{\sigma_i}$
- Gracias a esto, es más fácil encontrar un *learning rate* que funcione bien para todos los pesos. Si las variables tuvieran cada una una escala diferente, algunos $\eta$ serían demasiado bajos para algunas variables y muy altos para otras.