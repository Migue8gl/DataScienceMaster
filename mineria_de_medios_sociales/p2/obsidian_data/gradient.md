- #gradient | #gradiente 

### Introducción
- El gradiente de una función [[differentiable function|diferenciable]] escalar con múltiples variables es el [[vector]] $\triangledown f$, cuyos valores en cada punto $p$ representan la dirección y ratio de aumento de la función.
- Si el gradiente dado un punto es distinto de cero, la dirección del gradiente será aquella en la que la función aumente más rápido.
- La magnitud del gradiente es el ratio de aumento en esa dirección, el valor absoluto más grande de la derivada.
- Un punto donde el gradiente sea cero, se conoce como **punto estacionario**.

### Representación
- Cuando en un sistema de coordenadas los vectores base usados no son funciones de posición, es decir, estos vectores base no cambian según nos movemos por el sistema de coordenadas, el vector del gradiente se compone por [[derivatives#Derivadas parciales|derivadas parciales]] de $f$ en un punto $p$.
	- Para $f:\mathbb{R}^n\rightarrow\mathbb{R}$, su gradiente se define como $\triangledown f:\mathbb{R}^n\rightarrow\mathbb{R}^n$ definido en el punto $p=(x_1,...,x_n)$ como:
		- $\triangledown f(p)=\{\frac{\partial f}{\partial x_1},...,\frac{\partial f}{\partial x_n}\}$
		- Suele expresarse por columnas y no por filas.

### Derivada total vs gradiente
- Ua derivada total puede considerarse una transformación lineal que toma un vector de entrada (que representa pequeños cambios en las variables) y produce un vector de salida.
- El gradiente, por su parte, es un vector tangente. Proporciona información sobre la dirección y la magnitud del aumento más pronunciado de la función en un punto determinado. Representa el ratio instantáneo de cambio de la función a lo largo de cada eje de coordenadas.
- La derivada (derivada total) en un punto es un vector cotangente. Un vector cotangente es una función lineal que toma un vector como entrada y produce un escalar. En este caso, mide la tasa de cambio de la función en una dirección específica, dada por el vector de entrada.

### Gradiente disperso o "sparse"
- Un gradiente disperso, también conocido como un gradiente "sparse" en inglés, se refiere a un gradiente en el contexto del aprendizaje automático que contiene muchos elementos con un valor de cero. En otras palabras, solo una pequeña fracción de los elementos en el gradiente tiene valores diferentes de cero, mientras que el resto son cero.
- Suele darse en problemas de procesamiento del lenguaje natural.
