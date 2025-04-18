- #maths | #dl | #ml | #softmax

- La función **softmax** o softargmax convierte un vector de $N$ números reales en una distribución de probabilidad de $N$ salidas. Es una generalización de la función [[logistic function|logística]] para varias dimensiones.
- Antes de aplicar softmax, algunos componentes del vector podrían ser negativos, o mayores que uno; y podrían no sumar $1$; pero después de aplicar softmax, cada componente estará en el intervalo $[0,1]$, y los componentes sumarán $1$, por lo que pueden interpretarse como probabilidades. Además, los componentes de entrada más grandes corresponderán a probabilidades más grandes. 
- Función:
	- $\sigma(z)_i=\frac{e^{z_i}}{\sum_{j=1}^Ne^{z_j}}$
	- Donde $z$ es un vector de dimensión $N$.