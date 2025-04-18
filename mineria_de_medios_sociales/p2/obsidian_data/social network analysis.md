#socialnetworks | #datascience

# Introducción
- La minería de medios sociales tiene como objetivo principal **extraer, analizar y obtener conocimiento útil** a partir de los datos generados en plataformas de redes sociales.
- La mayoría de redes observadas en sistemas reales son **dispersas**, es decir, $L<<L_{max}$ o $k<<N-1$. Siendo $L$ el número de aristas y $L_{max}$ el número máximo de conexiones por nodo, es decir, no todos los nodos están conectados entre sí. $k=\frac{2L}{N}$, es decir, el grado medio de una red no dirigida, es decir, el promedio de conexiones por nodo en la red es mucho menos al número de nodos de la red.
- En ciencia de redes se usan medidas como:
	- *Distribución (de probabilidad) de grados* $p_{k}$: Probabilidad de encontrar nodos con $k$ enlaces.
	- *Distancia media de los caminos entre todos los nodos* : $d$.
	- *Coeficiente de Clustering* $C_{i}$: Mide la densidad local de la red, ¿qué proporción de los vecinos de cada nodo están conectados?
$$C_i=\frac{2L_{i}}{k_{i}(k_{i}-1)}$$
$$\langle C\rangle=\frac{1}{N}\sum_{i=1}^{N}C_{i}$$
- Donde $C_i$ es el coeficiente de *clustering* y $\langle C\rangle$ el coeficiente de *clustering* medio.

# Objetivos
- Identificar los actores más influyentes o centrales de la red mediante medidas estadísticas.
- Identificar *hubs* y autoridades, usando algoritmos de análisis de enlaces en redes dirigidas.
- Determinar patrones de interacción comunes entre actores, mediante medidas de niveles de interacción.
- Descubrir grupos de actores cohesionados, con técnicas de detección de comunidades.

# Medidas
## Medidas locales (nivel de actor)
- Todas ellas están basadas en el concepto general de **centralidad** (redes no dirigidas) o **prestigio** (redes dirigidas), una medida general de la posición de un actor en la estructura global de la red social. Se suelen usar para identificar actores clave.
- Los actores más centrales son aquellos nodos con más conexiones, tienen acceso a mayor número de nodos.
## Medidas globales (nivel de red)
- Proporcionan información más compacta que permite evaluar la estructura de la red, aportando información sobre fenómenos sociales subyacentes.
### Grado
- Número de enlaces que conectan a un nodo con otros (no dirigido).
- En redes dirigidas se define el prestigio de entrada o (*in-degree*) como **soporte** y el prestigio de salida (*out-degree*) como **influencia**.
### Intermediación
- Es una medida pensada para capturar la correduría. La correduría se refiere al papel de un nodo como **puente** entre diferentes partes de la red. Un nodo con alta correduría **controla o facilita el flujo de información** entre otros nodos que, de otro modo, estarían desconectados o tendrían que tomar rutas más largas para comunicarse.
$$C_{b}(i)=\sum_{j,k\in V(G)/i}g_{jk}(i)$$
- Donde $g_{jk}(i)$ es el número de caminos mínimos que conectan cualquier par de nodos $j-k$ y que incluten al actor $i$ para el que se calcula la intermediación.
- La intuición detrás de esta medida es, ver al actor en una posición más favorable en la medida en la que dicho actor se sitúe entre los caminos que otros nodos necesitan para realizar sus conexiones por los caminos más cortos.
- Estos actores se denominan también **porteros** (*gatekeepers*) porque tienden a controlar el flujo de información entre comunidades.
### Cercanía
- La cercanía es una medida que le da importancia a aquellos nodos que estén cerca de otros muchos nodos, a menor distancia con el centro mayor cercanía.
$$C_{c}(i)=\frac{1}{\sum_{j=1}^{g}d(i,j)}$$
### Excentricidad
- Se define como la distancia del camino mínimo más largo entre él y cualquier otro actor de la red.
$$E(i)=max_{j\in V(G)/i}d(i,j)$$
- Los actores de mayor excentricidad se denominan **actores periféricos**, los de menor valor forman el centro de la red. También se define la métrica de **centralidad de excentricidad** como la inversa de la excentricidad. 
- Es una medida que se usa menos, pero puede ayudar a detectar comportamientos extraños, actores con comportamientos muy diferentes al habitual.
### Centralidad de vector propio
- Se basa en que la centralidad de un nodo concreto depende de cómo de centrales sean sus vecinos (prominencia). La idea básica es que el poder y el estatus de un actor (ego) se define recursivamente a partir del poder y el estatus de sus vecinos (alters).
![[social network analysis-eigenvector-centrality.png]]

# Conceptos
## Red libre de escala
- Una red **libre de escalas** es aquella cuya distribución de grados (número de conexiones por nodo) sigue una ley de potencias:
$$p_{k}\thicksim k^{\gamma}$$
- Siendo $\gamma$ normalmente $2$ o $3$.
- Esto explica que hay muchos con poco grado $k$.
- Esto implica que no hay un grado típico, pocos nodos tienen muchísimas conexiones (los llamados **hubs**) mientras que la mayoría tienen pocas, la red es heterogénea.
- Muchos más nodos por debajo de la media que por encima. Pocos nodos con un grado significativamente mayor que la media (hubs). Sistema desequilibrado hacia la izquierda.
![[social network analysis-free-scale.png]]![[social network analysis-basic-anayslis.png]]
## Propiedad de mundos pequeños (small-world)
- Normalmente, se da que en redes reales muy grandes, pese a existir la posibilidad de que la distancia más larga (diámetro) sea $N-1$, normalmente estas redes tienen un diámetro ($d_{max}$) muy bajo, lo que hace que cualquier nodo esté cerca de otro a pocos pasos.

# Detección de comunidades
- [[social network community detection]]