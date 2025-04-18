- #bayesiannetwork | #practice

- **Explica brevemente los dos enfoques básicos para aprender la estructura de una red bayesiana. Pon un ejemplo de un algoritmo para cada uno de los enfoques describiendo de forma breve cómo funciona.**
-  Los dos enfoques básicos son *Tests de Independencia* y *Score + Procedimientos de Búsqueda*.
	- Los procedimientos de *Tests de Independencia* se basan en realizar pruebas de independencia condicional sobre los datos para determinar la estructura de la red. Se identifican relaciones de dependencia e independencia entre variables para construir el grafo.
		- **PC Algorithm**: El algoritmo **PC** comienza con un grafo completamente conectado y elimina aristas basándose en pruebas de independencia condicional. Luego, orienta las aristas siguiendo reglas para evitar ciclos y asegurar la dirección correcta de las relaciones causales.
	- El procedimiento de *Score + Búsqueda* consiste en definir una función de puntuación que mide la calidad de una estructura dada y usar un procedimiento de búsqueda para encontrar la mejor estructura.
		- **K2 Algorithm**: Sigue un enfoque *greedy* para encontrar la mejor estructura posible. El algoritmo asume que las variables están ordenadas de antemano y un número de padres máximo por nodo. 
		- Para cada variable $X_{i}$ se comienza sin ningún nodo padre y se van añadiendo padres a la lista si esto hace que el *score* mejore. Este proceso se repite para cada variable. Es necesario definir una función de evaluación.

- **A la hora de estimar una probabilidad condicionada, ¿qué diferencia hay entre la corrección de Laplace y un método bayesiano con un tamaño muestral equivalente S?**
- La corrección de *Laplace* agrega una cantidad pequeña $k$ a los conteos para evitar probabilidades nulas, generalmente $k=1$. Esto garantiza que ninguna probabilidad sea exactamente $0$ y mejora la estimación en presencia de valores poco frecuentes.
- El método bayesiano es una generalización de *Laplace*, en vez de sumar una corrección fija, introduce un **prior bayesiano** basado en una distribución previa sobre las probabilidades, con un parámetro $S$ que representa un "tamaño muestral equivalente", un tamaño fictio de observaciones previas.
- El enfoque bayesiano permite una mayor personalización en la estimación de probabilidades condicionadas, pero un método es solo una especialización del otro.

- **¿Qué métodos conoces para estimar probabilidades en una red bayesiana cuando hay datos perdidos? Describe brevemente sus principios.**
- Algoritmo *EM*: Se calculan las probabilidades esperadas de los valores faltantes basándose en los parámetros actuales. Se actualizan los parámetros de la red con base en la nueva información estimada. Se repite hasta la convergencia.
- Métodos variacionales: Siguen un enfoque bayesiano. Las probabilidades a "posteriori" no son fáciles de calcular y no son independientes. En lugar de calcular de forma exacta la distribución a posteriori, esta se aproxima con la mejor distribución de una familia. Para eso se utiliza la distancia Kuback-Leibler.

- **¿Están obligados los algoritmos de aprendizaje de redes bayesianas basados en evaluación y búsqueda a utilizar técnicas de búsqueda heurística?**
- En teoría se podría explorar el espacio completo, pero la complejidad de esto es exponencial con el número de variables, lo que hace inabordable el problema. Por ello es necesario utilizar métodos heurísticos.

- **¿Por qué es útil que las métricas empleadas en los algoritmos de aprendizaje de redes bayesianas tengan la propiedad de “descomponibilidad”?**
- Esto facilita la evaluación incremental durante la búsqueda, ya que al modificar la estructura solo es necesario recalcular la puntuación de las variables involucradas, reduciendo la complejidad computacional. Permite que la puntuación global se exprese como suma de puntuaciones locales.

- **¿Es cierto que los algoritmos de aprendizaje de redes bayesianas basados en evaluación y búsqueda emplean diferentes métricas y diferentes técnicas de búsqueda, pero siempre realizan la búsqueda en el espacio de grafos dirigidos acíclicos?**
- La red bayesiana se define formalmente como un *DAG*, por lo que la búsqueda de estructuras “válidas” implica garantizar la ausencia de ciclos. Sin embargo, algunas variantes optan por buscar en el espacio de clases de equivalencia de DAGs (por ejemplo, utilizando RPDAGs)

- **Enumera algunas métricas empleadas por los algoritmos de aprendizaje de redes bayesianas que estén basadas en ideas bayesianas y comenta brevemente la idea en la que se basan.**
- $K2$: Calcula la probabilidad a posteriori de la estructura dada la data, asumiendo un ordenamiento de las variables y utilizando una función de puntuación.

- **¿Por qué es importante para los algoritmos de aprendizaje de redes bayesianas basados en detección de independencias que los tests de independencia realizados sean del menor orden posible?**
- Los tests de independencia de alto orden (condicionados a muchos variables) requieren estimaciones basadas en subconjuntos de datos más pequeños, lo que aumenta la varianza y el riesgo de errores (falsos positivos o negativos). Al utilizar tests de bajo orden se mejora la robustez y la fiabilidad.

- **¿Cuáles son las diferencias principales entre el estimador bayesiano y el de máxima verosimilitud para la estimación de probabilidades a partir de datos?**
- El estimador de máxima verosimilitud se enfoca en encontrar los parámetros que maximizan la función de verosimilitud de los datos sin incorporar información previa.
- El estimador bayesiano combina la información de los datos con información a priori sobre los parámetros, obteniendo una distribución a posteriori.
- Uno considera información previa y el otro no.

- **¿Es cierto que cuando disponemos de pocos datos, las diferencias entre el estimador bayesiano y el de máxima verosimilitud no son importantes?**
- Con pocos datos, el *MLE* puede verse fuertemente afectado por la escasez de observaciones, generando estimaciones poco fiables (por ejemplo, probabilidades cero para eventos no observados). En contraste, el estimador bayesiano, al incorporar un prior (por ejemplo, a través de técnicas de suavizado como la corrección de Laplace), proporciona estimaciones más robustas y evita sobreajustes.