- #numericalstability

### Estabilidad en álgebra lineal
- Considerando el problema cuya solución depende de un algoritmo numérico, expresado como una función $f$ que mapea los datos $x$ a la solución $y$. El resultado del algoritmo/función es $y^*$, siendo este una solución aproximada de la solución real $y$.
- Las causas principales de esto son el **redondeo** (debido a limitaciones computacionales).
- Definimos dos tipos de error:
	- [[backward error]]
	- [[fordward error]]
- El algoritmo/función se dice *backward-stable* si el error **backward** es pequeño para todos los inputs $x$.
- La **estabilidad numérica**, por lo general, hace referencia a una estabilidad mezclada entre fordward y backward.