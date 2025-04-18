- #mh | #aco | #antcolony
- [wikipedia info](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)
- [heurística de deseabilidad o eta con SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
- **Paper**: *An advanced ACO algorithm for feature selection*.

- El algoritmo de colonia de hormigas o **ACO** es una técnica probabilistica que trata de resolver problemas computacionales que pueden ser reducidos a la búsqueda de caminos óptimos a través de grafos.
- Se basa en el comportamiento de las hormigas reales y su comunicación via feromonas.
- Las hormigas vagan por el mundo en búsqueda de comida de forma aleatoria. Cuando estas encuentran comida, dejan trazas de feromonas en su camino, de esta forma, si otras hormigas encuentran ese rastro, se hace más probable que recorran ese camino y dejen de vagar de forma aleatoria.
- Sin embargo, las trazas de feromonas se evaporan con el tiempo, perdiendo su fuerza de atracción sobre otras hormigas. De esta forma, si el camino es muy largo, la hormiga tardará mucho en recorrerlo y la traza de feromonas tendrá más tiempo a evaporarse. De manera análogamente inversa, a más corto es el camino, más rápido se recorre y más densidad de feromonas acumula.

### Representación
- Para poder adaptar el algoritmo ACO a un problema es necesario reducir ese problema a de la búsqueda del camino más corto dentro de un grafo ponderado.

### ACO Feature Selection
- Cada característica del problema original se representa como un nodo $n$. Los caminos que conectan a los nodos ($e$) representan la elección del subconjunto de características. El camino deberá ser lo más corto posible maximizando a su vez el **accuracy**.

#### Regla de transición probabilística
- $P_{ij}^k(t)=\begin{cases} \frac{\tau_{ij}^{\alpha}\eta_{ij}^{\beta}}{\sum_l\tau_{il}^{\alpha}\eta_{il}^{\beta}}&\hspace{6mm}\text{Si l y k son nodos admisibles} \\ 0 &\hspace{6mm}\text{De lo contrario}\end{cases}$
- $P_{ij}^k(t)$ denota la probabilidad de transición de un nodo de $i$ a $j$ en la $k$-hormiga (agente) en el instante de tiempo $t$. 
- $\tau_{ij}$ es la cantidad de traza de feromona en la arista $(i,j)$ en el momento $t$. $\eta_{ij}$ es la heurística de deseabilidad o visibilidad de la arista.
- $\beta$ y $\alpha$ son dos parámetros que controlan la importancia relativa del valor de la feromona vs la información de la heurística.
- El balance entre [[heuristic|heurística]] y feromona es el balance entre [[exploration&exploitation|explotación y exploración]].

- Después de que todas las hormigas hayan terminado su camino, la evaporación de feromonas comienza. El contenido de feromonas del camino $(i,j)$ en el instante $t+1$ es:
	- $\tau_{ij}(new)=(1-p)\tau_{ij}(t)+\sum_{k=1}^m\Delta\tau_{ij}^k(t)+\Delta\tau_{ij}^g(t)$
	- Donde:
		- $\Delta\tau_{ij}^k=\begin{cases}\frac{Q}{F^k}&\hspace{6mm}\text{Si la hormiga k pasa por la arista (i,j) en }T^k\\0 &\hspace{6mm}\text{De lo contrario}\end{cases}$
	- Donde $p\in (0,1]$ es el ritmo de evaporación, $m$ es el número de hormigas, $\Delta\tau_{ij}^k(t)$ y $\Delta\tau_{ij}^g(t)$ son respectivamente, la cantidad de feromonas colocadas en la arista $(i,j)$ por la hormiga $k$ y la cantidad de feromonas depositadas por la mejor hormiga $g$ en el instante $t$ sobre la arista $(i,j)$.
	- $Q$ es una constante y $F^k$ es el valor de coste de la solución encontrada por la hormiga $k$ en el tour $T_k$, es decir, $F^k$ es el [[fitness function|fitness]] de la hormiga $k$.
- Se utiliza el sistema *max-min*, solo la mejor hormiga puede depositar feromona.

#### TACO (Touring Ant Colony)
- **TACO** fue inicialmente diseñado para manejar variables continuas, las cuales eran decodificadas en string binarios.
- En este algoritmo cada solución es representada como un string de bits y solo se decidía utilizando la información de la feromona.

![[taco-example-feature-selection.png]]

#### bACO
- Este algoritmo usaba la estrategia *max-min*, permitiendo solo a la mejor hormiga depositar feromona y limitando el valor de esta entre un valor mínimo y uno máximo.

#### ACOFS
- Se utiliza el criterio [[f-score]] como valor heurístico (visibilidad de las aristas), pero con una estrategia de actualización de feromonas distinta. Cada nodo representa una carateríctica, cada nodo tiene además dos subnodos $0$ y $1$, deseleccionado y seleccionado. Se propone la siguiente ecuación:
	- $P_{01}(t)=\frac{\tau_{01}}{\tau_{01}+\tau_{00}}$
	- Donde $P_{01}$ es la probabilidad asociado al subcamino $(0\rightarrow 1)$, y $\tau_{00}$ y $tau_{01}$ son las feromonas artificiales de los subcaminos $(0\rightarrow 0, 0\rightarrow 1)$.

#### ABACO
- Combinación del tradicional **ACO** y **bACO**.
- No hay necesidad de predefinir el número de características a ser seleccionadas (limitación de **ACO**). Las características se redefinen como nodos completamente conectados entre sí ([[graph theory#Grafo completo $K_n$|grafo completo]]) (limitación de **bACO**).
- En cada iteración todas las hormigas visitan todos los nodos, pero pueden decidir si escoger la característica o no. La hormiga dentro del nodo $i$ debe escoger entre el subnodo $0$ o $1$, como en **bACO**. Así se van seleccionando o deseleccionando nodos hasta cubrir todos los nodos no visitados.
- Se define la regla de transición probabilística:
	- $P_{ix,jy}^k(t)=\begin{cases} \frac{\tau_{ix,jy}^{\alpha}\eta_{ix,jy}^{\beta}}{\sum_l\tau_{ix,l0}^{\alpha}\eta_{ix,l0}^{\beta}+\sum_l\tau_{ix,l1}^{\alpha}\eta_{ix,l1}^{\beta}}&\hspace{6mm}\text{Si l y k son nodos admisibles} \\ 0 &\hspace{6mm}\text{De lo contrario}\end{cases}$
	- Donde $x,y\in[0,1]$, donde $i$ es el índice del nodo actual (característica) de las $n$ totales, $j\in\text{nodos sin visitar}$. 
	- $k$ es la hormiga actual y $P$ su probabilidad de transicionar. $\tau$ es la feromona, $\eta$ la visibilidad heurística.