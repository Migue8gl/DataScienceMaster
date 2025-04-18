- #derivatives | #derivadas | #maths

# Concepto
- Una derivada nos da información sobre la pendiente de una [[functions|función]]. Esta es útil para medir el grado o ratio de cambio de una función en cualquier punto de esta en el espacio.
- Cuando la pendiente es $0$ significa que estamos en un mínimo o un máximo del valor de una función (o punto de inflexión). Esta información es muy importante en muchos ámbitos, como #machinelearning con el algoritmo del gradiente descendente.
- Pongamos de ejemplo la función $f(x)=x²$. Si queremos calcular la "inclinación" de la función en el punto $x=2$ podemos dibujar una línea tangente que solo "toque" a la función en ese punto exacto.
- Como para calcular una línea necesitamos dos puntos, esta se definirá en el punto $x$ que queramos y en el punto $x$ más una cantidad muy pequeña.
- Para calcular la inclinación entre dos puntos usamos la siguiente fórmula:
	- $m=\frac{y_2-y_1}{x_2-x_1}$
	- En nuestro caso $x_1=2,x_2=2.1$. En este caso $m=4.1$.
- A medida que hacemos ese pequeño paso entre $x_1$ y $x_2$ estaremos aproximando de forma más precisa el verdadero valor de la pendiente.
- Si aproximamos infinitamente obtenemos el valor de la pendiente para un punto dado, en nuestro caso, $m=4$.
- Cuando escribimos $\frac{d}{dx}$ estamos diciendo que derivamos respecto de $x$, es decir, el objetivo sobre el que queremos obtener la pendiente es la variable $x$.
## Ejemplo en Sympy
```python
from sympy import symbols, limit

s = symbols('s')
x = symbols('x')

f = ((x+s)**2 - x**2) / ((x+s) - x)
result = limit(f, s, 0)
```
- Lo que viene a decir es que si hacemos $s$, que es el paso o diferencia entre dos puntos, cada vez más pequeño, acercándose infinitamente a $0$, entonces obtendremos la derivada en $x$, que nos devuelve el valor de la pendiente para un punto $x$.
$$\large{f(x)=x^2}$$
$$\large{\frac{d}{dx}f(x)=\frac{(x+s)²-x²}{(x+s)-x}}$$

# Definición formal
- Formalmente, dada una función $f(x)$, la derivada en un punto $x$ se define como el límite: 
$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

# Definición Geométrica
- Para calcular la pendiente de la tangente a $f(x)$ en un punto $x = a$, se considera el cociente de diferencias entre dos puntos cercanos: $$m = \frac{f(x + h) - f(x)}{h},$$
- donde $h$ representa la distancia entre los dos puntos. Al hacer $h \to 0$, obtenemos la pendiente exacta: $$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}.$$

# Derivadas parciales
- Estas derivadas tienen como *input* funciones multivariables. En este caso estaríamos calculando la pendiente respecto a múltiples variables en varias direcciones en un espacio $n$-dimensional, donde $n$ es el número de variables de la función $f$.
- A este tipo de pendientes se les llama **gradientes**. El gradiente de un vector es un vector con todas sus derivadas parciales.
- Cuando se calculan derivadas parciales sobre una variable, el resto se fijan como si fuesen constantes.
- $f(x,y)=2x³+3y³$
- $\frac{d}{dy}f(x)=9y²$
- $\frac{d}{dx}f(x)=6x²$
