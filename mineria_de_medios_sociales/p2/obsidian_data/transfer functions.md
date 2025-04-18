- #transferfunctions | #v-shaped | #sigmoid | #s-shaped

- Una función de transferencia dentro del contexto de la búsqueda de soluciones para problemas de optimización es un tipo de función que traslada o asigna un espacio continuo a uno binario, es decir, lo mapea.
- De esta forma, puntos en el espacio con valores continuos/reales son obligados a clasificarse como $0$ o $1$.
- Existen varias familias, **s-shaped** y **v-shaped** (llamadas así por su forma en el [[cartesian plane|sistema cartesiano]]) son las más famosas.
- Las funciones **v-shaped** no fuerzan que los valores sean $0$ o $1$.

![[v-shaped-vs-s-shaped.png]]

## Sigmoidal

^bda699

- La función sigmoidal es un tipo de función de transferencia, más concretamente **s-shaped**.
- Fórmula:
	- $s = \frac{1}{1+e^{-x}}$
- Una forma de actualización de puntos en el espacio sería:
	- $x_{t+1}^d=\begin{cases}1 & \text{If } r_1 < s(x_{t+1}^d) \\0 & \text{If } r_1 \geq s(x_{t+1}^d)\end{cases}$
- Siendo $x$ un vector posición con dimensión $d$ en la iteración $t+1$. El valor $r_1$ es un número aleatorio entre $[0,1]$.
## Hiperbólica tangente
- Función de transferencia del tipo **v-shaped**.
- Fórmula:
	- $h = |tanh(x)|$
- Una forma de actualización de puntos en el espacio sería:
	- $x_{t+1}^d=\begin{cases}\lnot x_{t+1}^d & \text{If } r_1 < h(x_{t+1}^d) \\x_{t+1}^d & \text{If } r_1 \geq h(x_{t+1}^d)\end{cases}$
- Siendo $x$ un vector posición con dimensión $d$ en la iteración $t+1$. El valor $r_1$ es un número aleatorio entre $[0,1]$.