- #socialnetworks | #communitydetection

# Introducción
- Las redes complejas suelen mostrar estructuras de comunidades. Esta propiedad suele dar como consecuencia de la heterogeneidad global y local de la distribución de los enlaces en un grafo.
- Las comunidades se pueden definir como conjuntos de nodos similares, grupos de nodos conectados densamente entre sí y dispersamente netre grupos de la red.

# Modularidad
- La modularidad $Q$ es una función que mide la calidad de una partición concreta de una en comunidades. Toma un valor máximo (teórico) cuando la red presenta todos los enlaces dentro de cada comunidad y ninguno entre comunidades.
$$Q=\frac{1}{2\cdot L}\sum_{ij}[A_{ij}-\frac{k_{i}\cdot k_{j}}{2\cdot L}]\delta(c_{i},c_{j})$$
- Donde $L$ es el número total de enlaces de la red, $A_{ij}$ es una matriz de adyacencia, $k_{x}$ son los grados del nodo $x$, $\frac{k_{i}\cdot k_{j}}{2\cdot L}$ es la probabilidad esperada de que haya un enlace entre nodo $i$ y nodo $j$ (proporcional a sus grados) y $\delta(c_{i,}c_{j})$ es la función de *Kronecker* que vale $1$ si los nodos $i$ y $j$ pertenecen a la misma comunidad o $0$ en caso contrario.
- Cuando mayor es $Q$, mejor es la partición, es decir, las comunidades detectadas están densamente conectadas (más enlaces de los que cabría esperar aleatoriamente) y dispersamente conectadas entre sí. 
- Un valor $Q=0$ corresponde a un agrupamiento trivial. En la práctica $Q=0.3$ es un buen valor.
![[social network community detection-modularity-examples.png]]

# Formación de opiniones
- Si cada nodo adopta la opinión de la mayoría de sus vecinos, es posible formar opiniones distintas en subgrupos cohesivos diferentes. Suele haber más uniformidad dentro de un grupo cohesivo.

# Métodos de detección
- Se consideran cuatro familias de criterios no exclusivos:
	- **Comunidades centradas en nodos y en grupos de nodos**:
		- Cada nodo del grupo satisface ciertas propiedades (*criterio estructural*).
		- Se consideran las conexiones del grupo globalmente. El grupo entero debe satisfacer ciertas propiedades (*criterio estructural*).
	- **Comunidades centradas en jerarquía**: Se construye una estructura jerárquica de comunidades (*clustering jerárquico*).
	- **Comunidades centradas en la red**: Se divide la red completo en conjuntos disjuntos (*particionamiento de grafos*).
## Centradas en nodos y en grupos de nodos
### Identificación de cliques
- Todos los miembros del grupo tienen enlaces al resto. Los [[clique|cliques]] pueden estar solapados.
- La identificación de estas estructuras es un problema [[np-hard problem|NP-completo]]
- No son robustos, ya que un solo enlace faltante los desclasifica como comunidad.
- No son interesantes, ya que todo el mundo está conectado entre sí, no hay estructura centro-periferia, medidas de centralidad no dan información, no hay jerarquías entre enlaces, etc.
- Un solapamiento de cliques es más interesante.
### Comunidades fuertes y débiles
- Sea un subgrafo conexo $G_{s}$ con $N_{s}$ nodos.
- Se define el grado interno de un nodo $i$, $k_{i}^{int}$, como el conjunto de enlaces que lo conectan con otro nodo del mismo subgrafo.
- Se define el grado externo de un nodo $i$, $k_{i}^{ext}$, como el conjunto de enlaces que lo conectan al resto de la red.
- Si $k_{i}^{ext}=0$ entonces todos los vecinos de $i$ pertenecen al subgrafo y $G_{s}$ es una buena comunidad. Si $k_{i}^{int}=0$ todos los vecinos de $i$ pertenecen a otras comunidades e $i$ debe asignarse a una comunidad distinta.  
- Una **comunidad fuerte** (nodos) se refiere a que cada nodo $G_{s}$ tiene más enlaces dentro de la comunidad que con el resto del grafo.
$$k_{i}^{int}(G_{s})>k_{i}^{ext}(G_{s})$$
- Una **comunidad débil** (grupo) se refiere a que el grado interno total de $G_{s}$ es mayor que su grado externo total.
$$\sum\limits_{i\in G_{s}}k_{i}^{int}(G_{s})>\sum\limits_{i\in G_{s}}k_{i}^{ext}(G_{s})$$
![[social network community detection-strong-weak-community.png|450]]
## Centradas en red y jerarquía
- Existen dos líneas de investigación principales para el descubrimiento de comunidades a nivel global en redes complejas: **clustering jarárquico** y **particionamiento de grafos**.
- Estos métodos devuelven particiones disjuntas del conjunto de nodos, cada nodo pertenece a una única comunidad (no permiten el solapamiento de comunidades).
- Los métodos centrados en jearquía se basan en construir una estructura de este tipo. Hay dos enfoques principales $\rightarrow$ [[hierarchical clustering]]