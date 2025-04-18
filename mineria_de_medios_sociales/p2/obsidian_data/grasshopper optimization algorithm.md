- #mh | #grasshoper

## Inspiración
- Los **grasshoppers** (saltamontes o langosta cuando están en enjambre) son insectos cuyo ciclo de vida consiste en tres etapas: *huevo, ninfa, adulto*.
- Un saltamontes puede ser encontrado en el enjambre en su estado ninfa y adulto.
- La diferencia entre ninfa y adulto es la velocidad de movimiento con pasos pequeños en la ninfa y lo contrario en su vida adulta (Las soluciones *adultas* se encargan de la [[exploration&exploitation|exploración]] y las ninfas de la explotación).

## Algoritmo
### Posición
- $X_i=r_1S_i+r_2G_i+r_3A_i$
- Donde $X_i$ define el movimiento del saltamontes $i$. $S_i$ es la [[grasshopper optimization algorithm#Interacción social|interacción social]], $G_i$ es la [[grasshopper optimization algorithm#Fuerza de gravedad|fuerza de gravedad]], $A_i$ la [[grasshopper optimization algorithm#Advección al viento|advección del viento]] y $r_1, r_2, r_3$ son números aleatorios en el intervalo $[0,1]$. Todos los índices se refieren a la solución $i$.
- Esta función de actualización de posición no puede ser usada para la optimización debido a que se alcanza la zona de comfort de manera prematura y no se converge en un punto específico, por ello se utiliza la [[grasshopper optimization algorithm#Nueva posición|nueva fórmula]].

![[behaviour-GAO-opt-alg-with-first-position-formula.png]]

### Interacción social
- $S_i=\sum_{j=1,j\neq i}^N s(d_{ij})\hat{d_{ij}}$
- Donde $d_{ij}$ es la [[distances|distancia euclidiana]] entre el saltamontes $i$ y el $j$. $\hat{d_{ij}}$ es un [[vector]] unitario -> $\frac{x_j-x_i}{d_ij}$ desde el saltamontes $i$ al $j$.  $s$ es la [[grasshopper optimization algorithm#Fuerza social|fuerza social]] dada una distancia.

### Fuerza social
- $s(r)=fe^{\frac{-r}{l}}-e^{-r}$
- Donde $f$ indica la intensidad de atracción y $l$ es longitud de la escala de atracción.

![[social-forces-and-distance-interaction-grasshopers.png]]

- La repulsión entre saltamontes ocurre en el intervalo $[0,2.079]$. A partir de $2.079$ unidades de distancia no hay ni atracción ni repulsión entre individuos.
- Este rango es lo que se conoce como *zona de comfort*.
- Puede observarse en el gráfico que modificar las variables $l, f$ modifican las zonas de comfort, regiones de repulsión y regiones de atracción, de manera que puede modificarse el comportamiento de los saltamontes.
- La función $s$ puede dividir el espacio en estas tres regiones, pero para valores mayores a $10$ en cuanto a distancia, $s$ devuelve valores cercanos a cero, por lo que no es posible aplicar grandes fuerzas de atracción entre saltamontes separados por grandes distancias (por ello se mapea la distancia entre $[1,4]$).
### Fuerza de gravedad
- $G_i=-g·\hat{e_g}$
- Donde $g$ es la constante gravitacional y $\hat{e_g}$ es un vector unidad hacia el centro de la tierra.

### Advección al viento
- $A_i=u·\hat{e_w}$
- Donde $u$ es una constante de deriva y $\hat{e_w}$ un vector unitario con la dirección al viento.
- Los saltamontes en estado *ninfa* no tienen alas, por lo que su dirección de movimiento está muy marcada por este factor.

![[grasshoper-repulsion-attraction.png|500]]

### Nueva posición
- $X_i^d=c_1(\sum_{j=1,j\neq i}^N c_2\frac{ub_d-lb_d}{2}s(|x_j^d-x_i^d|)\frac{x_j-x_i}{d_ij})+\hat{T_d}$ 
- Donde $ud_b$ es la cota superior en la dimensión $D$, $lb_d$ es la cota inferior en la dimensión $D$, $\hat{T_d}$ es el valor de la dimensión $D$ en el objetivo (la mejor solución encontrada hasta la fecha) y $c$ es el coeficiente decreciente para hacer cada vez más pequeña la zona de comfort, zona de atracción y zona de repulsión.
- No se considera la gravedad ([[grasshopper optimization algorithm#Fuerza de gravedad|G]]) y la dirección del viento siempre es a favor del objetivo $\hat{T_d}$.
- Una diferencia con otros algoritmos de enjambres como [[particle swarn optimization|PSO]] es que **GAO** actualiza la posición de cada saltamontes a partir de la posición actual, el mejor punto encontrado hasta el momento y la posición de todos los otros agentes, mientras que **PSO** lo hace a partir de la posición actual del agente y la mejor posición encontrada.
- Esto significa que PSO no tienen en cuenta al resto de agentes, mientras que GAO sí.
- Los parámetros $c1, c2$ tienen dos propósitos:
	- $c1$ -> Reduce el movimiento del enjambre entero alrededor del objetivo, es decir, maneja la relación entre [[exploration&exploitation|exploración y explotación]] para el conjunto global de soluciones.
	- $c2$ -> Reduce la zona de comfort y las regiones de repulsión y atracción. $s$ es la función que decide si un agente debe sentirse atraído o repelido al objetivo, mientras que $\frac{ub_d-lb_d}{2}$ decrece de manera lineal el espacio de búsqueda a medida que $c_2$ decrece.
- Ha de tenerse en cuenta también que en el proceso de búsqueda, la exploración debe ir antes que la explotación. En los saltamontes sin embargo, la fase explotativa que es la **ninfa** es la que se da primero. Por ello el parámetro $c$ debe decrecer a medida que las iteraciones sobre el algoritmo van pasando. Este mecanismo hace que se promueva la explotación a medida que las iteraciones se incrementan, ya que se va reduciendo cada vez más el movimiento general del enjambre y se reduce la zona de comfort. $c$ se actualiza por tanto así:
	- $c=c_{max}-l·\frac{c_{max}-c_{min}}{L}$
- Donde $l$ es la iteración actual y $L$ el número máximo de iteraciones.

![[gao-optimization-graph-visualization-search.png|500]]

## Pseudocódigo

![[grasshoper-mh-pseudocode.png]]