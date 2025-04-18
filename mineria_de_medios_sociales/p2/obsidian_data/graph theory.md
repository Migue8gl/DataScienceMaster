- #graphtheory | #teoríadegrafos | #lmd

# Generalidades sobre los grafos 6.1
### Definición 1
- Un grafo $G$ es un par $(V,E)$, donde $V$ y $E$ son conjuntos, junto con una aplicación $\gamma g:E\rightarrow\{\{u,v\}:u,v\in V\}$.
- Al conjunto $V$ se le llama conjunto de vértices; al conjunto $E$ conjunto de lados o aristas, y a la aplicación $\gamma g$ (o simplemente $\gamma$) aplicación de incidencia.
- Si $e_1,e_2$ son dos lados tales que $\gamma g(e_1)=\gamma g(e_2)$ entonces se dicen que ambos lados son paralelos.
- Un lado tal que $\gamma g(e)=\{v\}$ se conoce como #lazo.

### Definición 2
- Un grafo dirigido u orientado es un par $(V,E)$, donde $V$ y $E$ son conjuntos, junto con dos aplicaciones $s,t:E\rightarrow V$.
- Las aplicaciones $s,t$ se conocen como aplicaciones *dominio* y *codominio*.

### Definición 3
- Sea $G=(V,E)$ un grafo  con aplicaciones de indidencia $\gamma g$. Un subgrafo de $G$ es un nuevo grafo $G^´=(V´,E´)$ donde $V´\subseteq V, E´\subseteq E$ y se verifica que $\gamma g´(e)=\gamma g(e)$ para cualquier $e\in E´$.
- Si $G´=(V´,E´)$ es un subgrafo de $G=(V,E)$, se dice que es un subgrafo completo si dado $e\in E$ tal que $\gamma g(e)\subseteq V´$, se verifica que $e\in E´$. Dicho de otra forma, si tiene todos los lados que tenía $G$ y que unen vértices de $V´$.

### Definición 4
- Sea $G$ un grafo. Un camino de longitud $n$ es una sucesión de lados $e_1e_2...e_n$, junto con una sucesión de vértices $v_0v1...v_n$ tales que $\gamma g(e_i)=\{v_i-1,v_i\}$.
- En tal caso se dice que el camino $e_1e_2...e_n$ es un camino del vértice $v_0$ al vértice $v_n$.
- De $v$ a $v$ se considera que hay un camino de longitud $0$, cuya sucesión de lados es vacía.
- Si estamos en un grafo sin lazos ni lados paralelos, cada lado está determinado por los vértices que une. En tal caso, para dar un camino daremos únicamente una sucesión de vértices.
- Un camino en el que no aparecen lados repetidos se llama #recorrido.
- Un **recorrido** en el que no hay vértices repetidos (salvo eventualmente el primero y el último) se conoce como #caminosimple.
- Un camino en el que coinciden el primer y último vértice se conoce como #caminocerrado.
- Un **recorrido** que a su vez es **camino cerrado** se llama #circuito. 
- Un **circuito** que a su vez es **camino simple** se llama #ciclo.

| Vértices repetidos | Aristas repetidas | Abierto | Nombre |
|:------------------:|:-----------------:|:-------:|:------:|
||||Camino|
|||No|Camino cerrado|
||No||Recorrido|
||No|No|Circuito|
|No|No||Camino simple|
|No|No|No|Ciclo|

### Proposición 6.1.1
- Sea un grafo $G$. Supongamos que existe un camino de $u$ a $v$. Entonces existe un camino simple de $u$ a $v$.

### Proposición 6.1.2
- Sea un grafo $G$ y sean $u,v$ dos vértices distintos. Supongamos que tenemos dos caminos simples distintos de $u$ a $v$. Entonces existe un ciclo en $G$.

### Definición 5
-  Sea $G$ un grafo. Se dice que $G$ es #conexo, si dados $u$ y $v$ dos vértices de $G$ existe al menos un camino de $u$ a $v$.
- Podemos definir la relación $uRv$ en el conjunto de vértices de la relación que cumple las propiedades de una [[equivalent relation|relación de equivalencia]].
- Cada uno de estos subgrafos es lo que se denomina *componente conexa de $G$*.

![[subgraph-component-related.png]]

### Definición 6
- Sea $G=(V,E)$ un grafo conexo, y $v,w\in V$. Se define la distancia de $v$ a $w$ como el mínimo de las longitudes que empiezan en $v$ y terminan en $w$. Ese número, será denotado como $d(v,w)$.

### Definición 7
- Un **grafo ponderado** es un grafo $G=(V,E)$ junto con una aplicación $p:E\rightarrow\mathbb{R⁺}$. Si $e$ es un lado del grafo, al número $p(e)$ se le conoce como peso del lado $e$.
- Si un grafo es ponderado se define la longitud de un camino como la suma de los pesos que componen dicho camino.

### Algoritmo de Dijkstra
- Este algoritmo nos devuelve el camino de longitud mínima entre un vértice y el resto.
- Supongamos que tenemos un grafos ponderado $G$ con $n$ vértices y sea $v$ un vértice de $G$. El algoritmo consta de $n$ iteraciones.
- Al inicio de  de cada iteración hay una serie de vértices marcados con un número y un vértice.
- En cada iteración:
	- Se elige el vértice marcado con el número más pequeño. Supongamos que este vértice es $u$, y que su marca es $(m_0,v_0)$. La marca de este vértice ya no se modifica en el resto.
	- Se actualizan las marcas de todos los vértices adyacentes al vértice $u$ y que no hayan sido elegidos previamente. Sea $w$ uno de estos vértices. La actualización se realiza como sigue:
		- Se calcula la suma de $m_0$ y el peso lado $uw$. Sea $m$ el valor de esta suma (es decir, $m=m_0+p(uw)$).
		- Si el vértice $w$ no está marcado, en esta iteración queda con la marca $(m,u)$.
		- Si el vértice $w$ está marcado, y su marca es $(m_1,v_1)$, se calcula el mínimo de $m$ y $m_1$. Si el mínimo es $m_1$ se queda con la marca que tenía, mientras que si es $m$ se le cambia la marca por $(m,u)$.
	- Una vez que se acaban las $n$ iteraciones, todos los vértices están marcados. La marca de cada vértice nos dice cuál es la distancia de este vértice al vértice $v$ y el penúltimo vértice del camino que une $v$ con este vértice (el último es el propio vértice).
	- Para iniciar el algoritmo se marca el vértice $v$ con $(0,v)$.

# Representación matricial de grafos 6.2
### Definición 8
- Sea $G$ un grafo cuyo conjunto de vértices es $V=\{v_1,v_2,...,v_n\}$. Se define su matriz de adyacencia como la matriz $A\in\mathbb{M}_n(\mathbb{N})$ cuyo coeficiente $(i,j)$ es igual al número de lados $e$ que unen $v_i$ con $v_j$.
- Esta matriz es **simétrica**.
- Un grafo puede tener varias matrices de adyacencia asociadas, cambian con la ordenación de los vértices. Si $A,C$ dos dos matrices de adyacencia de un mismo grafo, entonces existe $P$ tal que $P^{-1}CP=A$. Esta matriz es la matriz de permutación.
- La existencia de lados paralelos se traduce en coeficientes $(i,j)>1$. 
- La existencia de lazos se traduce en coeficientes $(i,j)=0$.
- Toda matriz cuadrada con coeficientes en $\mathbb{N}$ define un grafo.

![[adjacent-matrix-graph.png]]

### Proposición 6.2.1
- Sea $G$ un grafo cuyo conjunto de vértices es $\{v_1,v_2,...,v_n\}$ y sea $A$ su matriz de adyacencia. Entonces el coeficiente $(i,j)$ de la matriz $A^n$ es igual al número de caminos de longitud $n$ que unen $v_i$ con $v_j$.

### Definición 9
- Sea $G$ un grafo cuyo conjunto de vértices es $V=\{v_1,v_2,...,v_n\}$ y cuyo conjunto de lados es $E=\{e_1,e_2,...,e_n\}$. Se define la matriz de **incidencia** del grafo $G$ como una matriz $n\times m$ que tiene en la posición $(i,j)$ un $1$ si $v_i\in\gamma g(e_j)$ y $0$ en otro caso.
- Puede haber varias matrices de incidencia (depende de la ordenación), pero siempre se puede llegar a la misma matriz mediante operaciones elementales por filas y/o columnas.
- Si un grafo tiene lados paralelos, entonces la matriz tendrá dos columnas iguales. Si un grafo tiene lazos, entonces la matriz tendrá filas con un único coeficiente $1$.

# Isomorfismo de grafos 6.3
### Definición 10
- Sean $G=(V,E)$ y $G´=(V´,E´)$ dos grafos con aplicaciones de incidencia $\gamma g$ y $\gamma g´$. Se dice que $G$ y $G´$ son **isomorfos** si existen dos biyecciones $h_V:V\rightarrow V´$ y $h_E:E\rightarrow E´$ tales que para cualquier lado $e\in E$ se verifica que $\gamma g´(h_E(e))=\{h_V(u),h_V(v)\}$ donde $\{u,v\}=\gamma g(e)$.
- Tener dos grafos isomorfos es equivalente a dos matrices de adyacencia tales que dada una permutación $P$ ocurra que $P^{-1}CP=A$, siendo $A,C$ las dos matrices de adyacencia.
- Básicamente dos **grafos** son **isomorfos** si tienen el mismo número de vértices y los vértices de cada **grafo** se pueden numerar de 1 hasta n de modo que dos vértices del segundo **grafo** están unidos por una arista si y sólo si los dos vértices del primer **grafo** que tienen los mismos números están unidos por una arista.

### Definición 10
- Una propiedad se dice invariante por #isomorfismo si dados dos grafos isomorfos $G,G´,$ uno satisface la propiedad si, y sólo si, la satisface el otro.

### Definición 11
- Sea $G$ un grafo y $v$ un vértice de $G$. Se define el grado de $v$ como el número de lados (no lazos) de $G$ que son incidentes en $v$ más $2$ veces el número de lazos indicentes de $v$. Se denota como $gr(v)$.
- Denotaremos como $D_k(G)$ al número de vértices de $V$ que tienen grado igual a $k$. Podemos definir la sucesión $D_0(G),D_1(G),...,D_k(G),...$
- Si $G$ es un grafo con $n$ vértices y $l$ lados, la suma de todos los grados de cada vértices es igual a -> $2l$.

### Definición 12
- Sea $G$ un grafo (sin lazos ni lados paralelos). Se dice que $G$ es regular si todos los vértices de $G$ tienen el mismo grado.

# Algunas familias de grafos 6.4
## Grafo completo $K_n$
- Estos son grafos en los que cada vértice está unido mediante un lado a todos los vértices que forman el grafo. Sin lazos ni lados paralelos.

### Definición 13
- Se llama grafo completo de $n$ vértices al grafo que no tiene lazos ni lados paralelos, y dados dos vértices hay un lado que los une. Dicho de otra forma, su matriz de adyacencia toma el valor "cero" en todos los elementos de la diagonal y el valor "uno" en el resto. Dicho grafo se denota como $K_n$.
- El número vértices es $n$ y el de lados es -> $l=\frac{n(n-1)}{2}$.

![[complete-graph.png]]

## Grafo bipartido completo $K_{m,n}$
### Definición 14
- Sean $m,n$ dos números naturales distintos del "cero". Definimos el grafo $K_{m,n}$ como el grafo cuyo conjunto de vértices es $V=\{v_1,v_2,...,v_m,w_1,w_2,...,w_n\}$ y que para cada $i\in\{1,2,...,m\}$ y para cada $j\in\{1,2,...,n\}$ hay un lado que une los vértices $v_i$ con $w_j$. Además, no hay más lados aparte de estos.
- El número de vértices es $m+n$ y el número de lados es -> $l=m·n$.

![[complete-bipartite-graph.png]]

## Grafo camino $P_n$
### Definición 15
- Sean $n\geq 0$. Se define el grafo $P_n$ como el grafo cuyo conjunto de vértices es $\{v_1,v2,...,v_n\}$ y para cada $i\in\{1,2,...,n-1\}$ hay un lado que une $v_i$ con $v_{i+1}$.
- Se tienen $n+1$ vértices y $n$ lados.

![[path-graph.png]]

## Grafo ciclo $C_n$
### Definición 16
- Sea $n\geq 3$. Se define el grafo $C_n$ como el grafo cuyo conjunto de vértices es $\{v_0,v_1,v_2,...,v_n\}$ y para cada $i\in\{1,2,...,n-1\}$ hay un lado que une $v_i$ con $v_{i+1}$. Además hay un lado que une $v_1$ con $v_n$.
- El número de lados y de vértices es $n$.

![[ciclical-graph.png]]

## Grafo rueda $W_n$
### Definición 17
- Sea $n\geq 3$. Se define el grafo $W_n$ como el grafo cuyo conjunto de vértices es $\{v_0,v_1,v_2,...,v_n\}$ y para cada $i\in\{1,2,...,n-1\}$ hay un lado que une $v_i$ con $v_{i+1}$. Además hay un lado que une $v_1$ con $v_n$, y para cada $i\in\{1,2,...,n\}$ hay un lado que une $v_0$ con $v_i$.
- El número de vértices es $n+1$ y el número de lados es $2n$.

![[wheel-graph.png]]

## Grafo cubo $Q_n$
### Definición 18
- Sea $n\geq 1$. Definimos el grafo $Q_n$ como el grafo cuyo conjunto de vértices es $\{(x_1x_2...x_n):xi\in\{0,1\}\}$ (es decir, el conjunto de vértices es el conjunto de cadenas de $n$ bits) y dos vértices $(x_1x_2...x_n)$ e $(y_1y_2...y_n)$ están unidos por un lado si se diferencian únicamente por un bit.

![[cubic-graph.png]]

# Sucesiones gráficas 6.5
- Sea $G$ un grafo sin lazos ni lados paralelos, cuyo conjunto de vértices es $\{v_1,v_2,...,v_n\}$. Para cada $i$ entre $1$ y $n$, sea $d_i=gr(v_i)$. Tenemos de esta forma una secuencia $d_1,d_2,...,d_n,$ que se corresponde con los grados de los vértices del grafo $G$.
- Nos planteamos si dada una lista de $n$ números naturales $d_1,d_2,...,d_n,$ existe algún grafo (sin lazos ni lados paralelos) con $n$ vértices tal que los grados de esos vértices sean estos números naturales. Y en caso de que exista, cómo podríamos dar un grafo con tales características. Antes de responder a estas cuestiones, vamos a introducir un poco de nomenclatura.


![[graph-example.png]]

- Este grafo, por ejemplo, tiene $gr(v_1)=gr(v_2)=4,gr(v_0)=gr(v_3)=2,gr(v_7)=gr(v_6)=gr(v_5)=gr(v_4)=1$. Por tanto la sucesión es $4,4,2,2,1,1,1,1$ es una **sucesión gráfica**.

### Definición 19
- Sean $d_1,d_2,...,d_n\in\mathbb{N}$. Decimos que la sucesión $d_1,d_2,...,d_n$ es una sucesión gráfica si existe un grafo $G$ de lazos ni lados paralelos, con conjunto de vértices $V=\{v_1,v_2,...,v_n\}$ y tal que $d_i=gr(v_i)$.
- Si $d_1,d_2,...,d_n$ es una sucesión gráfica, y $G$ es un grafo con $n$ vértices cuyos grados son los términos de la sucesión, diremos que $G$ es una realización de la sucesión $d_1,d_2,...,d_n$.

![[graphic-sucesion.png]]

### Teorema 6.5.1
- (Havel-Hakini). Sea $d_1,d_2,...,d_n$ una sucesión de números naturales. Supongamos que están ordenados en orden decreciente, es decir, $d_1\geq d_2\geq ...\geq d_n$ y que $d_1<n$. Entonces esta sucesión es gráfica si, y sólo si, lo es la sucesión $d_2-1,...,d_{d_1+1}-1,d_{d_1+2},...,d_n$.
- La condición $d_1<n$ nos dice que no puede haber un vértice cuya grado sea mayor o igual al número de vértices.
- Si la suma de la sucesión es impar, también sabemos que no es una sucesión gráfica.

### Ejemplo
- Consideramos la sucesión $2,2,2,2,$ que sabemos que es gráfica. Comprobamos con el método descrito:
	- Están ordenados. Eliminamos el primer elemento y restamos $1$ a los dos siguientes elementos:
		- $1,1,2$.
	- Reordenamos:
		- $2,1,1$.
	- Volvemos a repetir:
		- $0,0$.
	- Como nos queda todo "ceros", sabemos que es **gráfica**.
- Consideramos ahora la sucesión $5,4,4,2,2,1$.
	- $5,4,4,2,2,1$ -> 
	- $3,3,1,1,0$ ->
	- $2,0,0,0$ ->
	- $-1,-1,0$.
- No es sucesión gráfica pues no puede haber vértices con grado $-1$.

### Algoritmo de reconstrucción
- La idea es ir recorriendo las distintas sucesiones que nos han aparecido, pero al revés, y en cada paso ir construyendo un grafo en el que se materialice la sucesión correspondiente.

### Ejemplo
- Consideramos el primer ejemplo del [[graph theory#Ejemplo|ejemplo anterior]].
- Tenemos la sucesión $0,0$:

![[reconstruction-graph-1.png]]

- Después tenemos la sucesión $2,1,1$, vemos que hay que añadir aristas y un vértice:

![[reconstruction-graph-2.png]]

- Por último nos queda la sucesión original $2,2,2,2$:

![[reconstruction-graph-3.png]]

# Grafos de Euler 6.6
### Definición 20 
- Sea $G$ un [[graph theory#Definición 5|grafo conexo]]. Un camino de Euler es un recorrido en el que aparecen todos los lados.
- Un circuito de #euler es un camino de Euler cerrado.
- Un grafo con un circuito de Euler es un grafo de Euler.

### Proposición 6.6.1
- Sea $G$ un grafo. Si $G$ es un grafo de Euler, el grado de cada vértice es par, mientras que si G tiene un camino de Euler, $G$ tiene exáctamente dos vértices de grado impar (donde empieza y acaba el camino).

### Teorema 6.6.1
- Sea $G$ un grafo conexo. Entonces $G$ es un grafo de Euler si, y sólo si, el grado de cada vértice es par.

### Lema 6.6.1
- Sea un grafo $G$ (con un número finito de vértices) en el que cada vértice tiene grado mayor que $1$. Entonces $G$ contiene un circuito (y por tanto un ciclo).

### Corolario 6.6.1
- Sea $G$ un grafo conexo. Entonces $G$ tiene un camino de Euler si, y sólo si, $G$ tiene exactamente dos vértices de grado impar.

### Algoritmo de Fleury
- Este algoritmo calcula un camino de Euler de un grafo que sabemos que lo tiene.
- Como entrada tenemos un grafo $G$. Como salida, dos sucesiones $S_V$ y $S_E$, que son las sucesiones de vértices y lados del camino buscado.
	- **1** - Si todos los vértices son de grado par, elegimos un vértice cualquiera $v$. Si $G$ tiene dos vértices de grado impar elegimos uno de estos vértices.
	- **2** - Hacemos $S_V=v$ y $S_E=[]$.
	- **3** - Si $G$ tiene sólo a $v$, devuelve $S_V$ y $S_E$, y termina.
	- **4** - Si hay un único lado $e$ que incida en $v$, llamamos $w$ al otro vértice donde incida el lado $e$. Quitamos de $G$ el vértice $v$ y el lado $e$ y vamos al paso **6**.
	- **5** - Si hay más de un lado $e$ que incida en $v$, elegimos uno de forma que al quitarlo del grafo $G$ siga siendo [[graph theory#Definición 5|conexo]]. Llamamos $e$ a dicho lado y $w$ al otro vértice en el que indice $e$.
	- **6** - Añadimos $w$ al final de $S_V$ y $e$ al final de $S_E$.
	- **7** - Cambiamos $v$ por $w$ y volvemos al paso **3**.

# Grafos de Hamilton 6.7
### Definición 21 
- Sea $G$ un grafo. Un camino de Hamilton es un camino que recorre todos los vértices una sola vez. 
- Un circuito de Hamilton es un [[graph theory#Definición 4|camino cerrado]] que recorre todos los vértices una sola vez (salvo extremos).
- Un grafo con un circuito de Hamilton se denomina circuito hamiltoniano.
- Si hay algún vértice de grado $1$, entonces no es de Hamilton. Además si tiene $n$ vértices, si es de Hamilton, debe tener mínimo $n$ lados.

### Teorema 6.7.1
- Sea $G$ un grafo con $n$ vértices.
- Si el número de lados es mayor o igual que $\frac{1}{2}(n-1)(n-2)+2$, entonces el grafo es hamiltoniano.
- Si $n\geq 3$ y para cada par de vértices no adyacentes se verifica que $gr(v)+gr(w)\geq n$, entonces $G$ es un grafo de Hamilton.

# Grafos Bipartitos 6.8
### Definición 22
- Sea $G=\{V,E\}$ un grafo. Se dice que $G$ es bipartito o bipartido si podemos descomponer $V$ en dos subconjuntos disjuntos $V_1, V_2$ de forma que cada lado incide en un vértice de $V_1$ y en un vértice de $V_2$.

### Teorema 6.8.1
- Sea $G=(V,E)$ un grafo. Entonces $G$ es bipartito si, y sólo si, $G$ no contiene ciclos de longitud impar.

### Lema 6.8.1
- Sea un grafo $G$ bipartito con partición del conjunto de vértices $V=V_1\cup V_2$. Supongamos que $v_1v_2...v_m$ es un camino en $G$ y que $v_1\in V_1$. Entonces $\{v_1,v_3,v_5,...\}\subseteq V_1$ y $\{v_2,v_4,...\}\subseteq V_2$.

### Proposición 6.8.1
- Sea $G$ un grafo bipartito con partición $V_1$ y $V_2$. Supongamos que $|V_1|=n$ y $|V_2|= m$. Entonces:
	- Si $G$ tiene un [[graph theory#Grafos de Hamilton 6.7|camino de Hamilton]], entonces $|n-m|\leq 1$.
	- Si $G$ es un grafo de Hamilton, entonces $n=m$.
	- Si $G$ es [[graph theory#Grafo bipartido completo $K_{m,n}$|completo]] y $|n-m|\leq 1$, entonces $G$ tiene un camino de Hamilton.
	- Si $G$ es completo y $n=m$, entonces $G$ es un grafo de Hamilton.

# Grafos planos 6.9
### Definición 23 
- Sea $G$ un grafo. Una representación de $G$ se dice plana si los vértices y los lados se encuentran todos en un plano, y las líneas que representan dos lados distintos no se cortan. Un grafo se dice plano si acepta una representación plana.

![[plane-graph.png]]

### Teorema 6.9.1
- (Característica de Euler). Sea $G$ un grafo plano y conexo. Llamemos $v$ al número de vértices, $l$ al número de lados y $c$ al número de caras de una representación plana. Entonces $v-l+c=2$. En general, si $G$ es un grafo plano, y $X$ es el número de componentes conexas entonces $v-l+c=1+X$.

### Corolario 6.9.1
- En un poliedro, si $v$ es el número de vértices; $l$ es el número de aristas y $c$ es el número de caras entonces $v-l+c=2$.

### Corolario 6.9.2
- Sea $G$ un grafo plano, conexo, sin lazos ni lados paralelos y sin vértices de grado $1$. Entonces $3c\leq 2l$ y $l\leq 3v-6$.

### Definición 24
- Sea $G=(V,E)$ un grafo (sin lazos ni lados paralelos). Sea $v$ y $w$ dos vértices adyacentes y sea $e$ el lado que los une. La contracción simple de $G$ a través de $e$ es el grafo siguiente:
	- El conjunto de vértices es $V\backslash\{w\}$ (eliminamos el vértices $w$).
	- Para formar el conjunto de lados, eliminamos el lado $e$. Y cada lado que una al vértice $u$ lo sustituimos por un lado que una $u$ con $v$ (siempre y cuando ese lado no exista en el grafo $G$).
- Una contracción de $G$ es una cadena de contracciones simples.

![[contraction-graph-example.png]]

### Teorema 6.9.2
- (Kuratowski). Sea $G$ un grafo. Entonces $G$ es plano si, y sólo si, ningún subgrafo suyo puede contraerse a $K_5$ ni a $K_{3,3}$.

### Definición 25
- Sea $G$ un grafo plano. Supongamos que tenemos una representación plana con caras $c_1,c_2,...,c_r$. Definimos el grafo *dual* para la representación dada como el grafo cuyo conjunto de vértices es igual al conjunto de caras (o tiene un vértice $v_{i}^{'}$ para cada cara $c_i$), y cuyo conjunto de lados coincide (o es biyectivo) con el conjunto de lados de $G$. En el grafo dual, un lado une dos vértices si en la representación plana de $G$ de dicho lado es frontera común de las dos caras.

![[dual-graph.png|500]]

# Coloración de grafos 6.10
### Definición 26
- Sea $G=(V,E)$ un grafo. Una coloración $G$ es una aplicación $f:V\rightarrow C$, donde $C$ es un conjunto, de tal forma que para cualquier $e\in E$, si $\gamma_g(e)=\{e,w\}$ con $v\neq w$ entonces $f(u)\neq f(v)$.
- Se llama **número cromático** de $G$ al cardinal del menor conjunto $C$ (colores) para el que existe una coloración de $G$, y lo denominamos $X(G)$.
- En general, para un grafo $K_n$ su número cromático es $n$.
- Para un grafo bipartito, su número cromático es $2$.
- Si $G_1$ es un subgrafo de $G_2$, entonces $X(G_1)\leq X(G_2)$.
- Si un grafo es plano, su número cromático es menor o igual a $4$.

### Definición 27
- Sea $G$ un grafo y $x\in\mathbb{N}$. Vamos a denotar por $p(G,x)$ al número de coloraciones distintas, con $x$ colores, que tiene el grafo $G$.
- $p(K_n,x)=x(x-1)...(x-n+1) = x^{\underset{-}{n}}$. 
- $p(G,x)=p(G_1,x)·p(G_2,x)·...·p(G_n,x)$. Siendo $G_i$ sus componentes conexas.
- $p(P_n,x)=x(x-1)^{n-1}$ 
- [[graph theory#Algunas familias de grafos 6.4|Familias de grafos]].
- A partir del polinomio cromático, si empezamos a probar con valores para $x$, entonces el primero que sea $\geq$ que $0$ será el **número cromático**. 

### Teorema 6.10.1
- Sea $G$ un grafo, y $u,v$ dos vértices adyacentes. Sea $e$ el lado que los une. Entonces $p(G_e,x)=p(G,x) + p(G_{e}^{'},x)$.
- Deducimos pues que $p(G,x)=p(G_e,x)-p(G_{e}^{'},x)$.

![[example-cromatic-pol-algorithm.png]]

# Árboles 6.11
## Caracterización de los árboles 6.11.1
### Definición 28
- Un árbol es un grafo conexo que no tiene ciclos. Un grafo que no tenga ciclos se denomina bosque (podría no ser conexo).
- Dado un grafo conexo, un subgrafo suyo se dice **árbol generador** si tiene todos los vértices y es un árbol.
- Un árbol no tiene lazos ni lados paralelos.

### Lema 6.11.1
- Sea un grafo $G$ conexo que tiene un ciclo. Entonces, si quitamos uno de los lados del ciclo el grafo sigue siendo conexo.

### Proposición 6.11.1
- Todo árbol es un grafo plano.

### Corolario 6.11.1
- Sea $G$ un grafo conexo con $n$ vértices. Entonces $G$ es un árbol si, y sólo si, $G$ tiene $n-1$ lados.

### Teorema 6.11.1
- Sea $G$ un grafo con $n$ vértices, sin lados paralelos ni lazos. Entonces son equivalentes:
	- $G$ es un árbol.
	- Dos vértices cualesquiera están unidos por un único camino simple.
	- $G$ es conexo, pero si le quitamos un lado deja de serlo.
	- $G$ no tiene ciclos, pero si le añadimos un lado tendrá algún ciclo.
	- $G$ tiene $n-1$ lados.
- Los árboles son los menores grafos conexos o los mayores grafos sin ciclos.

## Árboles generadores 6.11.2
- Dado un grafo $G=(V,E)$, un grafo generador es un subgrafo de $G$, que contiene todos los vértices, y que es un árbol.
- Todo grafo conexo tiene un árbol generador.
- Si a todos los lados del grafo le damos peso uno y aplicamos [[graph theory#Algoritmo de Dijkstra|el algoritmo de Dijkstra]] obtenemos un árbol generador.
- Si tenemos un grafo conexo podemos hacer lo siguiente:

### Cutting Down (destructiva)
- Comprobamos si $G$ tiene ciclos.
	- Si no tiene ciclos, ya tenemos un árbol generador.
	- Si tiene ciclos, quitamos un lado de ese ciclo y volvemos a empezar.
- Al terminar el proceso, tendremos un árbol generador.
- Se descartan $l − (n − 1) = l − n + 1$ lados de uno en uno de forma que cada uno de los que se descartan rompa un ciclo (forme parte de un ciclo) del grafo que va quedando.

### Building Up (constructiva)
- En primer lugar ordenamos todos los lados del grafo $e_1,e_2,...,e_n$.
- Comenzamos con el grafo $G_0=(V,E_0)$, donde $E_0=\emptyset$.
- Supuesto construido el grafo $G_i=(V,E_i)$ pasamos al grafo $G_{i+1}=(V,E_{i+1})$. 
	- Si al añadir $e_{i+1}$ al grafo $G_i$ el grafo resultante tiene algún ciclo, entonces $G_{i+1}=G_i$. 
	- Si no, entonces $G_{i+1}=(V,E_i\cup\{e_{i+1}\})$ (es decir, le añadimos el lado $e_{i+1}$) al grafo $G_i$.
- El grafo $G_n$ es un árbol generador.
- Básicamente se eligen $n − 1$ lados de uno en uno de forma que cada uno no forme ciclos con los anteriormente elegidos

### Algoritmo de Kruskal
- Para obtener un árbol generador mínimo (mínimo respecto al grafo ponderado del que se parte).
- Se ordenan los lados en orden creciente según sus pesos, en cuyo caso estamos utilizando -> Kruskal [[graph theory#Building Up (constructiva)|constructivo]].

![[kruskal-buildingup.png|500]]

- Se ordenan los lados en orden decreciente según sus pesos, en cuyo caso estamos utilizando -> Kruskal [[graph theory#Cutting Down (destructiva)|destructivo]].

![[kruskal-cuttingdown.png|500]]

### Algoritmo de Prim
- Se parte de un vértice arbitrario $v$ que pasa al conjunto de los elegidos $T=\{v\}$ y $E=\{\}$.
- En cada paso se añade un vértice $u$ a $T$ y un lado $e$ a $E$ con las condiciones:
	- Que $u$ sea un vértice que no esté en $T$.
	- Que $u$ sea adyacente mediante el lado $e$ a uno de $T$.
	- Que el lado $e$ no forme ciclo con los lados de $E$.
	- Que $e$ sea de peso mínimo entre los que complen las condiciones anteriores.
- El proceso se acaba cuando se eligen $n-1$ lados.

## Árboles con raíz 6.11.3
- Un árbol con raíz no es mmás que un árbol del que se ha destacado un vértice. Si $G$ es un árbol en el que hemos destacado como raíz al vértice $v_0$, hablaremos del árbol $(G,v_0)$. 
- Sea $(G,v_0)$ un árbol raíz:
	- La profundidad de un vértice $v$ es la longitud del único camino simple que une $v$ con la raíz. En tal caso, la profundidad de la raíz vale $0$.
	- La profundidad del árbol es el máximo de las profundidades de sus vértices.
	- Un vértice $v$ se dice que es una hoja si no es la raíz y su grado es igual a $1$.
	- Un vértice $v$ se dice que es hijo de $w$ si el único camino simple que une $v$ con la raíz es $vw...v_0$. Notemos que en tal caso, la altura de $v$ es una unidad mayor que la altura de $w$.
	- Un vértice de $w$ se dice que es padre de $v$ si $v$ es hijo de $w$.
	- Un vértice $w$ es descendiente de un vértice $v$ si $v$ está en el único camino simple que une $w$ con la raíz.
	- La altura de un nodo es la profundidad del subárbol formado por el nodo y sus descendientes.
	- La altura de un árbol es el mayor de los niveles de sus nodos.
	- El rango de un nodo es unca cualidad del nodo que sirve para compararlo con los que tiene en el mismo nivel, a más a la izquierda esté el nodo, menor rango.
- Nivel = Profundidad.
- Un árbol con raíz se dice *m-ario* si cada nodo tiene a lo sumo *m* hijos.

### Recorridos en árboles
- Para recorrer un árbol con raíz ha de tenerse en cuenta el **rango** de un nodo, la **profundidad** de un nodo y la **altura** de un nodo.

### Recorrido preorden
- Primero se recorre la raíz y después se recorren los subárboles hijos en preorden en orden creciente del rango de sus raíces.

### Recorrido postorden
- Primero se recorren los subárboles hijos en postorden en orden creciente del rango de sus raíces y después se recorre la raíz.

### Recorrido inorden
- Primero se recorre el subárbol hijo de menor rango en inorden, después la raíz, y por último se recorren en inorden el resto de subárboles hijos en orden creciente del rango de sus raíces.

### Recorrido top-down
- Se recorren los nodos en orden creciente de profundidad o nivel, y dentro de un nivel, en orden creciente de rangos. Es el recorrido en anchura usual y es trivial en el sentido de que es el orden de lectura de izquierda a derecha y de arriba a abajo.

### Recorrido bottom-up
- Es el más complejo. Se recorren los nodos en orden creciente de altura, dentro de los de misma altura en orden creciente de profundidad y dentro de los de la misma altura y profundidad en orden creciente de rango.

## Árboles etiquetados 6.11.4
### Definición 29
- Un árbol etiquetado es un árbol con $n$ vértices en que los vértices tienen como etiquetas los números naturales $\{1,2,...,n\}$.
- Dos árboles etiquetados son isomorfos si tienen el mismo número de vértices y la aplicación identidad es un isomorfismo de grafos.

### Teorema 6.11.4
- El número de árboles etiquetados con $n$ vértices es $n^{n-2}$.

### Código de Prüfer
- Suponemos los árboles etiquetados con los números naturales de $1$ a $n$. Llamamos código de Prüfer de un árbol etiquetado a la sucesión de longitud $n-2$ obtenida de la siguiente forma:
	- Se parte de la sucesión vacía (código vacío) y del árbol etiquetado $T$.
	- Si $T$ tiene dos vértices se devuelve el código y fin.
	- Se determina la hoja con menor etiqueta del árbol $T$, se añade al código la etiqueta del adyacente a la hoja seleccionada y $T$ pasa a ser el árbol con la hoja seleccionada suprimida.
	- Se vuelve al segundo paso con el nuevo código y el nuevo árbol

![[prufer-code.png]]

![[prufer-algorithm.png]]