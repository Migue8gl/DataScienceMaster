- #functions

### Definición
- Las funciones son expresiones que relacionan dos o más variables entre sí. Una función coge variables de entrada (variables de **dominio** o independientes), las evalúa en una expresión y devuelve el **output** o salida (variable dependiente).
	- $y=2x+1$.
	- $f(x)=2x+1$.
- Esta función de ejemplo es lo que se conoce como [[continuous function|función continua]].
- Las funciones bidimensionales suelen representarse en un [[cartesian plane|plano cartesiano]].
- Usando la librería de *Sympy* de Python (álgebra simbólica) podemos pintar por pantalla funciones:

```python
from sympy import *

x = symbols('x')
f = 2*x + 1
plot(f) 
```

