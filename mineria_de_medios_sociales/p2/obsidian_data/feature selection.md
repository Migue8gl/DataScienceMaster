- #ml | #featureselection | #optimizationproblem

# Introducción
## Definición
- El problema de la selección de características se define como el proceso de seleccionar un subconjunto de características relevantes.
- Una característica (**feature**) es una propiedad individual medible o característica de un fenómeno concreto.
- El problema de escoger el subconjunto óptimo de características es un [[np-hard problem|problema NP duro]].

## ¿Por qué?
- Se utiliza por varias razones:
	- Simplificación del modelo de forma que sea más fácil de interpretar.
	- Entrenamientos más cortos.
	- Para evitar [[curse of dimensionality|maldición de la dimensionalidad]].
- La premisa central nos dice que la selección de características se usa debido a que existen algunas de estas características que son redundantes o irrelevantes, de forma que no perdemos información.

# Métodos
- **All subsets/best subset**: El método más directo (y ineficaz) es el de probar todas las combinaciones posibles de parámetros y elegir el subconjunto que minimice algún criterio. El problema es que hay $2^p$ combinaciones posibles, por lo que no es viable.
- **Fordward selection**: Se comienza con el modelo **nulo**, es decir, aquel que solo contiene el intercepto ($\beta_0$) y ningún predictor.
	- Se ajustan $p$ modelos y se elige el que minimice el [[residual standard error|RSS]].
	- Se añade a ese mejor modelo la variable que resulte en el menor **RSS** de entre todos los modelos de dos variables.
	- Se continua hasta que algún criterio de parada se alcance.
	- **null** > **all 1-model** > **all 2-model** ...
- **Backward selection**: Se comienza con un modelo que tenga todas las variables.
	- Se elimina la variable con el [[p-values|p-valor]] mayor.
	- Se ajusta el nuevo modelo con $p-1$ variables.
	- Se elimina la variable con el p-valor mayor.
	- Se continua hasta alcanzar algún criterio de parada.