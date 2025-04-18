- #crossover | #ga | #blx | #spx | #mpx
- [wikipedia info](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm))

- El crossover o recombinación es un operador genético usado en algoritmos evolutivos tales como los [[genetic algorithm|GA]]. Es una forma estocástica de generar nuevos individuos en la población de soluciones, tal como funciona su análogo natural, la reproducción sexual.
- Para garantizar el incremento de la función fitness (función objetivo) solo los mejores individuos de la población son seleccionados para aparearse, de forma que las mejores soluciones son recombinadas entre sí para formas nuevos [[genetic algorithm#Genotipo y Fenotipo|genotipos]].
- Esta selección es la encargada de la parte fundamental de [[exploration&exploitation|explotación]] del algoritmo, mientras que el crossover o la [[mutation|mutación]] son operadores explorativos.

### BLX (Blend Crossover)
- Tipo de cruce que utiliza dos padres para crear dos descendientes. Se suele usar para representaciones codificadas en **reales**. Dado dos números reales para cada uno de los genes de los padres al hijo se le asignará un número aleatorio entre ese rango de gen para cada gen que conforme al [[vector]] cromosómico.

![[blx-crossover.png|400]]

- Características:
	- Distribución de probabilidad uniforme con una cota controlada por $\alpha$.
	- Diversidad en los hijos proporcional a la de los padres.
	- Búsqueda demasiado amplia si los padres están distantes.

### One-point crossover
- Se elige al azar un punto en los cromosomas de ambos progenitores y se designa como "punto de cruce". Los bits a la derecha de ese punto se intercambian entre los dos cromosomas parentales. El resultado son dos descendientes, cada uno con información genética de ambos progenitores.
- Se suele utilizar en GAs de codificación **binaria**.

![[one-poin-crossover.png]]

### K-point crossover
- En el cruce de *k* puntos, se eligen al azar *k* puntos de cruce de los cromosomas parentales. Los bits entre esos puntos de cruce se intercambian entre los organismos progenitores. 

![[k-point-crossover.png]]