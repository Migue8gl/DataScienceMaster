#statistics | #ml | #autoregresion

# Concepto
- La auto-regresión es un concepto muy usado en estadística, series temporales y *machine learning*. 
- El término se refiere a que el valor de una variable en un modelo puede ser predicho basandose en los valores de variables previas.
- El término "autorregresivo" refleja la idea de que la variable retrocede sobre sus propios valores pasados.
- Se puede representar como:
	- $X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + \epsilon_t$
	- Donde:
		- $X_t$ es el valor de la variable en el tiempo $t$.
		- $c$ es una constante.
	    - $\phi_1, \phi_2, \ldots, \phi_p$ son los coeficientes autorregresivos.
	    - $\epsilon_t$ es ruido blanco, que representa fluctuaciones aleatorias que no pueden ser explicadas por el modelo.