- #limits

### Significado
- Tomemos de ejemplo la [[functions|función]] $f(x)=\frac{1}{x}$ cuya gráfica se muestra tal que así:

![[1-divide-x-plot.png|450]]

- Según decrecen los valores para $x$, la función $f(x)$ se aproxima a $0$, pero nunca alcanza cero.
- Teniendo en cuenta la intuición de que algo puede estar infinitamente acercándose a un valor, se introducen pues los límites:
	- $\lim_{x\to\infty}\frac{1}{x}=0$
- A medida que $x$ se aproxime al infinito, la función se aproximará a $0$ (nunca alcanza cero).

### Ejemplo en Sympy

```python
from sympy import *

x = symbols('x')
f = 1/x
result = limit(f, x, 'oo')
```