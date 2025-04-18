- #tfg | #featureselection | #metaheuristics

- [[tfg script|guion]] 

## Notas
- Escribir al profesor para preguntar sobre implementación de ACO. Mandar explicación con el paper y la propuesta, preguntar sobre uso de heurística de deseabildiad y mandar enlace a proyecto ACO en github que usa SelectKbest con ANOVA F-Value.

## Links de interés
- [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance)
- [F-test](https://en.wikipedia.org/wiki/F-test)
- [Prueba KNN k](https://stackoverflow.com/questions/11568897/value-of-k-in-k-nearest-neighbor-algorithm)
- [Selección de modelos](https://en.wikipedia.org/wiki/Model_selection)
- [Escalado Z-Score y MinMax juntos](https://www.quora.com/Can-we-use-both-Z-score-standardization-and-Min-Max-scaling-as-data-input-for-machine-learning-models)
- [Wilcoxon test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

## Conceptos básicos
- [[abstract]] 
- [[survey]]
- [[combinatorial optimization]] 
- [[feature selection]]
- [[fitness function]]
- [[metaheuristic]]
- [[exploration&exploitation]]
- [[genetic algorithm]]
- [[simulated annealing]]
- [[greedy randomized adaptive search procedure]]
- [[tabu search]]
- [[ant colony algorithm]]
- [[whale optimization algorithm]]
- [[bat algorithm]]
- [[artificial bee colony optimization]]
- [[grey wolf optimizer]]
- [[dragonfly algorithm]]
- [[grasshopper optimization algorithm]]
- [[particle swarn optimization]]
- [[wrapper and filter based]]
- [[transfer functions]]
- [[no free lunch theorem]]
- [[k-nearest neighbors]]
- [[support vector machine]]

## Lista metaheurísticas con versión binaria
- binary Firefly Algorithm
- binary Whale Optimization Algorithm
- binary Bat Swarm Optimizer **(revisar)**
- binary Grey Wolf Optimizer
- binary Dragonfly algorithm **(revisar)**
- binary Grasshoper algorithm
- bACO
- bABC
- bPSO
- GA
- DE

## Escritura
- Frases cortas y claras.
- Figuras y tablas numeradas y referenciadas en el texto.
- Si hay figuras de otros, indicar fuente.
- Estructura:
	- Resumen:
		- En español y en inglés.
		- Contenido y objetivos principales del trabajo.
	- Introducción:
		- Objetivo general.
		- ¿Buenos en real son buenos en binario y viceversa?
		- Comparación objetiva.
		- Selección de características y su complejidad. Contexto dentro de la rama de la inteligencia artificial y aprendizaje automático.
		- Explicación del problema un poco más en profundidad.
		- Descripción y breve historia de las metaheurísticas y su aplicación en campos de optimización.
		- Planificación temporal y presupuesto.
			- Diagrama de Gantt.
			- Coste hardware, software, desplazamiento, tiempo.
	- Estado del arte / trabajos relacionados / contexto:
		- Revisión exhaustiva de la literatura existente sobre selección de características.
		- Descripción de métodos usados (filter methods, wrapper methods, PCA, etc).
		- Resumen de estudios previos.
	- Propuesta:
		- Herramientas usadas.
		- Diagrama de software.
		- Detalles de la metodología propuesta.
		- Algoritmos clasificatorios usados.
		- Descripción de las metaheurísticas seleccionadas y justificación de su elección.
		- Explicación del diseño experimental, se incluye conjuntos de datos utilizados, métricas de evaluación, configuración de los algoritmos.
	- Resultados y discusión:
		- Comparativas entre versiones continuas y versiones binarias.
		- Comparativas entre mejor versión binaria y mejor versión real.
		- Test estadísticos.
		- Tablas de métricas, gráficos, discusión y análisis exhaustivo de los distintos resultados.
		- Discusión sobre posibles mejoras o extensiones de cara al futuro.
	- Bibliografía.

## Planificación de tareas
- **Investigación inicial**
- **Diseño del software**
- **Investigación metaheurísticas**
- **Implementación del software**
- **Pruebas y refactorizado**
- **Análisis de resultados**
- **Documentación**

## Fundamentos teóricos
- Optimización.
	- Metaheurísticas.
- Complejidad y computabilidad. ? Teoría de la computación.
- Aprendizaje automático.
	- SVC y KNN.
- Selección de características.

## Propuesta
- Diseño software -> patrones de diseño utilizados, librerías, lenguajes, modularización del código y función de cada parte.
- Software de las metaheurísticas -> cita del repositorio, modificación de los algoritmos a binarios, tablas con los parámetros de cada metaheurística.
- Datasets -> tabla con los datasets usados.
- Diagrama software.

## Análisis
- Comparamos resultados usando los reales, y los analizamos.
- Comparamos resultados usando los binarios, y los analizamos.
- ¿Buenos en binario lo son en real?
- ¿Cómo son los clásicos frente a los modernos?
- ¿Recientes más prometedores?
	- Gráficas de convergencia (unas pocas)
	- Gráficas boxplot (unas pocas)
	- Rankings (todos)

## Correcciones documento
- Tabla con horas aproximadas y reales para cada tarea.
- Horas de cómputo en el servidor de investigación.
- Marcar en negrita tablas.

	https://tacolab.danimolina.net/

- En general empezar con ranking promedio.
- Variabilidad y diversidad (sección a parte) con boxplot.
	- Mencion a gráfica (un par de figuras).
- De forma general, empieza con figuras y con mejores y peores.
- Sistematico, si hablas de variabilidad, no se habla más luego.
- Boxplot en una sola figura.
- Tablas con estadísticas siempre lo último.
-  Los más destacables en cuanto al número de veces que estos se repiten por ser los mejores por dataset son bCS, ACO, bDE y bGWO.
	- Poner quizá tabla con esos algoritmos y cuantas veces son los mejores en los datasets???
- Particionar tablas de p-values -> excel -> latex
- El peor de todos los algoritmos comparados es, sin embargo, bABCO, un algoritmo clásico -> no repetirse y ponerlo en peor.
- Tablas del mejor vs resto -> poner qué algoritmo.
- Hipótesis nula y expl en diseño experimental.
- Comparar con test estadísticos con kNN y SVC por separado.
- Todas las tablas numericas al apéndice.
- Simplificar tiempo de ejecución, no hace falta test estadísticos. Poner tabla de tiempos promedios y ya. Mayoría de tiempos similares y dos notables excepciones, bABCO y bFA.
- En convergencia no poner: Claramente, el peor algoritmo y al que más le cuesta converger es bDummy. Pese a ello, bDummy termina por superar en algunos problemas a ACO, que según que problema parece variar mucho en su proceso optimizatorio.
- Última sección es binario vs continuo.

- Poner graficas de convergencia leyenda en mayusculas
- Página 98 referencias ??