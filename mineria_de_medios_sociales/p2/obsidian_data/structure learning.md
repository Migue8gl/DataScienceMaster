- #bayesiannetwork | #structurelearning

# Concepto
- *Structure learning* o aprendizaje estructural es la tarea de aprender un *DAG* (grafo dirigido acíclico) por medio de los datos. Hay dos aproximaciones a este problema, los basados en *score* y los basados en tests de independencia.

# Relaciones independientes
## PC Algorithm
### Hipótesis
- Las relaciones independientes tiene una representación perfecta en un **DAG**.
- Tenemos un *dataset* enorme.
- Los test estadísticos no tienen errores.
- Bajo esas condiciones se descubre una red bayesiana.
### Test estadísticos
- El algoritmo se basa en preguntar si la independencia verdadera de las relaciones es de la forma:
$$I(X_{i},X_{j}|A)$$
- Donde $A$ es un subconjunto de variables.
### Distancia KL
- [[distances|KL]].
### Cross Entropy
- [[cross entropy]].
### Test de independencia
- Para evaluar si $X$ e $Y$ son condicionalmente independientes dado $A$, se computa la entropía cruzada $CE(X,Y|A)$ donde las probabilidades son [[maximum likelihood estimation|los estimadores de máxima verosimilitud]] de la base de datos (frecuencias relativas).
- El estadístico del test es $G^2$ que es $2mCE(X,Y|A)$ donde $m$ es el tamaño de la muestra.
- Bajo asumpción de independencia, $G^2$ sigue una distribución $X^2$ con grados de libertad igual a $(r_{X}-1)(r_{Y}-1)\prod_{Z\in A}r_{Z}$. Donde $r_{W}$ es el número de variables de $W$.
- Rechazar el test implica decir que hay evidencia de que las variables tienen relaciones de dependencia entre ellas. Aceptar NO significa que sean independientes.
- A mayor número de variables se condicionan ($x\in A$) menor fiabilidad de los test. Igual ocurre si el tamaño de la muestra es pequeño.
### Algorithm
- Encontrar un patrón de grafo (un grafo no dirigido).
- Encontrar enlaces *head-to-head* evaluando independencias. Los enlaces cabeza a cabeza son dos nodos que tienen un hijo en común pero que no están conectados entre sí.
- Orientar los enlaces sin crear ciclos.
### Condición básica
- Dos nodos $X$ e $Y$ están conectados sí y solo sí no existe un subconjunto $S_{XY}$ de un conjunto de vértices $V$ tal que $I(X,Y|S_{XY})$.
![[structure learning-i=0-pc.png|400]]

# Score + búsqueda
- Determinan una puntuación que mide como de bien describe la red bayesiana los datos, y al mismo tiempo, añade una penalización a la complejidad del modelo.
## Score de verosimilitud
- Si se considera un *dataset* $D$ y un grafo $G$ con parámetros $\theta$, entonces se puede computar la verosimilitud de un grafo y sus parámetros dado los datos.
$$L(G,\theta:D)=\prod_{i=1}^{n}\prod_{j=1}^{q_{i}}\prod_{k=1}^{r_{i}}\theta_{ijk}^{N_{ijk}}$$
- Se suele considerar el logarítmo de la verosimilitud.
## Penalización de máxima verosimilitud
- Tiene la forma de:
$$\prod_{i=1}^{n}\prod_{j=1}^{q_{i}}\prod_{k=1}^{r_{i}}N_{ijk}log(\frac{N_{ijk}}{N_{ij}})-Pen(N)Dim(G)$$
- Donde $Dim(S)$ es el tamaño del modelo.
- $Pen(N)$ es una penalización (no negativa) del tamaño de la muestra $N$.
	- $Pen(N)=1$ -> *Akaike Information Criterion*.
	- $Pen(N)=\frac{1}{2}log(N)$ -> *Bayesian Information Criterion*.