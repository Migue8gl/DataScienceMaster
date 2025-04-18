- #ga | #genetic | #tfg
- [wikipedia info](https://en.wikipedia.org/wiki/Genetic_algorithm)

- Los algoritmos genéticos están inspirados en la selección natural y estos se utilizan tanto en problemas de optimización con restricciones como sin ellas.
- El algortimo genético es una [[metaheuristic|metaheurística]] que modifica una población de soluciones individuales de manera repetida de forma que, en cada paso, se seleccionan soluciones **padre** que dan lugar a parte de la siguiente generación de soluciones en la siguiente iteración del algoritmo.
- Las soluciones o **fenotipos** están compuestas por una serie de propiedades / características / cromosomas / **genotipos**, que pueden ser mutados y alterados. La forma más tradicional de representación es la de $0$ y $1$, es decir, la binaria, pero otros tipos de codificación son posibles.
- El proceso es el siguiente:
	- Se inicializa la población de manera aleatoria.
	- En cada iteración se crea una nueva población:
		- Se califica cada individuo/solución calculando el valor **fitness**.
		- Se escalan estas calificaciones para convertirlas en un rango de valores más útil. Estos valores escalados se llaman **valores de expectación**.
		- Se seleccionan miembros de la población basándonos en estos valores. Estos serán los padres (**SELECCIÓN**).
		- Se selecciona un grupo **élite** que pasará a la siguiente generación.
		- Se producen los **hijos**. Los hijos son producidos a partir de cambios aleatorios o distintos tipos de **cruce** en los que puede o no haber una mutación aleatoria.
		- Se reemplaza a la población actual con los hijos producidos.
		- El algoritmo para cuando sobrepasamos cierta tolerancia de error o ciertas iteraciones máximas.
- El algoritmo genético crea varios tipos de **hijos**:
	- *Élite* -> aquellos que tienen una evaluación fitness muy alta y por tanto sobreviven para pasar a la siguiente generación.
	- *Crossover* -> aquellos que se crean combinando parte del [[vector]] de ambos padres.
	- *Mutación* -> aquellos que se crean en alguna de las etapas anteriores y obtienen una mutación aleatoria.

### Crossover
- [[crossover]]

### Mutación
- [[mutation]]

### Elitismo
- Una práctica común en el proceso de generación de una nueva población es la del *elitismo*. Esta se basa en la selección de los mejores organismos de la población actual para pasar a la siguiente. 
- Es una estrategia que garantiza que la solución obtenida por el GA no decremente su calidad a medida que pasan las generaciones y además ayuda en la fase de [[exploration&exploitation|explotación]], por lo que el parámetro que controla el número de individuos *élite* que pasan a la siguiente generación debe ser escogido con de manera precisa.

### Genotipo y Fenotipo
- En el contexto biológico el *genotipo* es el conjunto de genes y la información genética que conforman a un individuo de cualquier especie. El *fenotipo* es la expresión en forma física de las características de un individuo.
- En el contexto de las ciencias de la computación, el *genotipo* es el conjunto de bits que representan una característica $x$, mientras que el *fenotipo* es la propia característica $x$ tras interpretar la información genética.
- Esta transferencia ocurre tras pasar por una función de mapeo.

![[phenotype-genotype-example.png|450]]

### Ejemplo práctico
- La siguiente imagen es la de una antena de la nave espacial ST5 de la NASA, cuya forma fue diseñada y optimizada por un algoritmo genético, maximizando el diagrama radiactivo.

![[antenna-ga-example.png]]