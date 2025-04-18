- #tfg | #guion

- Total en pruebas: **17 min**

## Introducción (6min)
- Buenos días, soy Miguel García y hoy os vengo a presentar mi proyecto: "Estudio y Análisis de Metaheurísticas Modernas para el Problema de Selección de Características".  Este TFG ha sido supervisado por el profesor Daniel Molina Cabrera.
- En esta presentación se seguirá el mismo orden que el descrito en el documento de la memoria. Iniciaremos con una contextualización del problema, la motivación del trabajo y los objetivos a cumplir. Una vez terminada la introducción procederé a describir como se ha llevado a cabo la planificación del proyecto, se realizará una revisión superficial de la literatura y más tarde se explicará cómo se han llevado a cabo los experimentos. Se finalizará exponiendo los resultados y unas conclusiones finales.
### Contexto
- La selección de características es un tipo de procesamiento que se aplica a grandes conjuntos de datos con numerosas características. Este implica elegir un subconjunto de características más relevantes para el problema que se esté tratando.
- Es un problema NP duro, por lo que pese a existir una forma de resolverse de manera exacta, el número de combinaciones posible escala de forma exponencial, y con ello el tiempo de resolución.
- Las ventajas de aplicar una selección de características son muchas, entre ellas algunas como reducción de ruido dentro del conjunto de datos, mejora de generalización del modelo, decremento de tiempo de entrenamiento, disminución de complejidad del modelo, mayor interpretabilidad, etc.
### Motivación 
- Este trabajo se eligió por varios motivos:
	- Importancia de este procesamiento en el campo del aprendizaje automático.
	- Dentro de las metaheurísticas, que es el tipo de algoritmo que hemos elegido, no hay gran cantidad de comparaciones entre ellas, menos entre versiones binarias y continuas. Además de la poca rigurosidad entre algunas de ellas.
- En este trabajo se han identificado las propuestas más novedosas y se han implementado las versiones binarias de estos.
- Para comparar todas las propuestas, se emplean distintas métricas como la precisión, el porcentaje de reducción de características, la función fitness, que es una mezcla de ambas métricas y tiempo de ejecución entre otras. Además se emplearán tests estadísticos robustos para poder realizar comparaciones de alta fiabilidad.
### Scopus
- En esta diapositiva, podemos observar un gráfico que ilustra la evolución de las publicaciones relacionadas con la selección de características a lo largo del tiempo. Lo que vemos aquí es revelador.
- La selección de características se ha convertido en un problema cada vez más estudiado y relevante en la comunidad científica. El gráfico muestra una clara tendencia al alza en el número de publicaciones sobre este tema.
- Lo mismo ocurre cuando buscamos metaheurísticas aplicadas al feature selection, la tendencia es igualmente alcista con el paso de los años.
### Objetivos
- Con este trabajo se pretende evaluar el desempeño de las metaheurísticas seleccionadas, investigar profundamente las diferencias entre versiones en codificación continua y codificación binaria. Se mostrarán las fortalezas y debilidades de las metaheurísticas seleccionadas y con ello se harán una serie de recomendaciones prácticas que varían con el contexto del problema a resolver.
- Por último se quiere evaluar los resultados obtenidos para poder dar respuesta a una serie de preguntas que veremos más adelante.
### Metaheurísticas
- Comencemos definiendo que son los algoritmos metaheurísticos y por qué son interesantes para aplicarlos a este tipo de problemas.
- Las metaheurísticas son algoritmos de optimización, es decir, su objetivo principal es el de minimizar o maximizar una función objetivo. En este caso, minimizar una función fitness.
- Son algoritmos normalmente bioinspirados, es decir, la construcción de estos suele estar fuertemente basada en comportamientos naturales tanto animales, como de comportamiento social o incluso basados en leyes físicas.
- El principal atractivo de estas es que son capaces de obtener resultados muy buenos en tiempos muy bajos. En problemas cuya resolución depende de un algoritmo exacto cuyo tiempo de ejecución no es viable, este tipo de algoritmos son muy interesantes.
- Concretamente, los algoritmos seleccionados pertenecen a la familia de algoritmos metaheurísticos poblacionales. Es así porque cada una de las posibles soluciones se representan como un individuo de la población, la cual va evolucionando y cambiando generación tras generación hasta obtener un individuo que represente la mejor solución posible.

## Planificación (1min y 30s)
- La planificación seguida consta de las siguientes tareas:
	- Investigación inicial.
	- Diseño del software.
	- Investigación de las metaheurísticas.
	- Implementación del software.
	- Pruebas y refactorizado.
	- Análisis de resultados.
	- Documentación.
- Aquí podemos observar el diagrama de Gantt elaborado. Algunas tareas requerían de un solapamiento, como son las de investigación de metaheurísticas y desarrollo software.
- Aquí tenemos una estimación del presupuesto, con unos 25 euros la hora, un ordenador cuyo uso tras 8 meses da un resultado de 133 euros y el uso de un servidor para ejecutar los experimentos.
- En total unos 8700 euros.

## Revisión de la literatura (2min)
- Para este proyecto se han escogido una serie de algoritmos basandonos en aquellos más novedosos, con mejor rendimiento según las comparaciones existentes y más citados. Este grupo es el que llamaremos modernos.
- Además se han incluido una serie de algoritmos clásicos, cuya robustez y buenos resultados han sido demostrados durante los años. Estos han sido aplicados a numerosos problemas. Serán los clásicos.
- La revisión será muy superficial debido a la gran cantidad de algoritmos a comparar, para más información se recomienda leer la sección dedicada a estos algoritmos de la memoria.
- Los algoritmos seleccionados son:
	- El algoritmo de los lobos grises, basado en su jerarquía social y comportamiento de caza.
	- El algoritmo de los saltamontes, inspirado en su movimiento e interacción entre ellos.
	- El de la luciérnaga, basado en la atracción que produce su brillo.
	- Búsqueda cuco, caracterizado por el comportamiento parasitario del pájaro y en los vuelos Levy.
	- Algoritmos genéticos, inspirados en la evolución genética.
	- Algoritmo de la ballena, inspirado en los comportamientos de caza de la ballena jorobada.
	- Colonia de abejas artificiales, basadas en una jerarquía de abejas y su búsqueda de alimentos.
	- Algoritmo de la libélula, caracterizados por sus operadores de cohesión y distanciamiento de enemigos.
	- Colonia de hormigas, que construyen caminos óptimos basandose en el rastro de feromonas.
	- Opimización por enjambre de partículas, que se basan en posiciones relativas y globales para su búsqueda del óptimo.
	- Algoritmo del murciélago, basado en la ecolocalización de estos.
	- Evolución diferencial, que hace uso de la operación de diferencia de vectores solución.

## Diseño experimental (1min y 30s)
- Se procede a explicar como se han realizado los experimentos.
### Conjunto de datos
- Se han seleccionado los siguientes conjuntos de datos por diferentes motivos:
	- Variedad de las áreas que estos tratan.
	- Número de características, se seleccionan algunos conjuntos con bastantes características, ya que el motivo principal de este proyecto es ver cuan capaces son los algoritmos de reducirlas.
	- Por su relevancia práctica, son algoritmos muy usados en la comunidad científica y por ello muchos de ellos constituyen un estándar.
### Fitness
- La función objetivo que se intenta minimizar está compuesta a su vez por dos métricas:
	- La precisión, es decir, el porcentaje de instancias bien clasificadas.
	- La reducción de características, es decir, el porcentaje de características descartadas.
- Se le da mucha más importancia a la precisión debido a que si se le diese demasiada ponderación a la reducción, los algoritmos perderían mucha calidad y tenderían a descartar características sin criterio. Con un 10% es suficiente.
### Parámetros
Los parámetros fijados en cada uno de los algoritmos han sido seleccionados de las recomendaciones de los autores en los artículos originales de cada algoritmo.

## Resultados (9min)
- A continuación procedo a explicar los resultados obtenidos en los experimentos. Me gustaría recordar que los resultados aquí expuestos son una pequeña parte de todas las comparaciones realizadas, las cuales pueden verse en la sección de análisis de la memoria y en el apéndice.
### Preguntas
- En este trabajo nos hemos planteado una serie de preguntas a investigar.
	- ¿Merece la pena el uso de versiones específicas de algoritmos para la selección de características?
	- ¿Cómo rinden los clásicos frente a los más recientes?
	- ¿Los algoritmos buenos en binario lo son también en su versión real?
	- ¿Cuáles son las opciones más interesantes?
- A todas estas preguntas se les dará respuesta en esta sección.
### Continuos
- Pasamos a los resultados obtenidos para los algoritmos continuos.
- En esta gráfica que representa el ranking de los algoritmos en versión continua usando el clasificador KNN, podemos observar visualmente los mejores y peores algoritmos en promedio.
- Esta gráfica muestra lo mismo, pero para SVC.
- Los resultados obtenidos son los siguientes:
	- En fitness hay dos claros ganadores, GWO y CS, mientras que los peores son GOA y DA.
	- Los algoritmos que mejor reducen, los que descartan más características, son CS y GWO, además con muchísima diferencia con respecto al resto.
	- El algoritmo más rápido es FA y el más lento ABCO. El resto tienen un tiempo de ejecución promedio muy similar.
- Aquí podemos ver la convergencia de los algoritmos para el problema de ionosphere con KNN, se puede observar que GWO y CS, junto a WOA, convergen mucho más rápido que el resto. Lo mismo ocurre en este otro conjunto de datos.
### Binario
- Aquí de nuevo muestro rankings para las versiones en binario tanto para KNN como SVC. 
- Los resultados son los siguientes:
	- En fitness vuelve a repetir GWO como el mejor, seguido de PSO. Mientras que GOA y ABCO dan los peores resultados,
	- En cuanto a reducción de características, el mejor con absoluto superioridad es el ACO.
	- Los más lentos y rápidos vuelven a ser los mismos, FA y ABCO.
### ¿Versión binaria mejor en feature selecction?
- ¿Es recomendable usar una versión específica de un algoritmo para la selección de características?
- Pues los resultados dicen que sí, la mayoría de algoritmos rinden mejor o igual en sus versiones binarias con el añadido de reducir el conjunto de características seleccionado. En la gráfica se ve el promedio de selección de cada algoritmo, siendo las versiones binarias las que menos características seleccionan.
- Los continuos también parecen capaces de reducir, pero los binarios son mejores a igualdad de precisión.
### ¿Recientes vs clásicos?
- Algoritmos recientes como CS y GWO suponen una mejora muy significativa con respecto a algoritmos clásicos.
- Pese a ello, algoritmos como GA, DE o PSO siguen siendo mejores que muchas opciones modernas en ambas versiones.
- Por tanto, algunos recientes ofrecen mejoras que no pasan desapercibidas, pero los clásicos no son una opción que pueda descartarse dados sus muy buenos resultados.
### Algoritmos recientes prometedores
- Dados los resultados en continuo y en binario, en los que el mejor indiscutible es GWO, se ve que claramente este algoritmo tiene una gran adaptabilidad, no solo entre distintos problemas, si no también entre distintas versiones.
### ¿Bueno en binario también en continuo?
- Con respecto a si un algoritmo es bueno en una versión, también lo es en la otra, los resultados parecen respaldar que sí. En esta tabla se muestran los valores fitness promedio entre algoritmos binarios y continuos, y sus valores son similares. Por lo tanto, si un algoritmo es bueno en una versión, es de esperar que su versión continua también lo sea. Lo mismo ocurre con aquellos algoritmos que no rinden bien.