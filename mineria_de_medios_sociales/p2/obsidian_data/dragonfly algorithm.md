- #mh | #dragonfly

## Inspiración
- Los **dragonflies** (libélulas en español) son pequeños insectos que se pueden ver en enjambre solo cuando cazan o migran.
- Las diferencias dadas entre ambos enjambres son notorias. En caza, se organizan en enjambres pequeos que vuelan en muchas direcciones en busca de comida. Este tipo de enjambre es llamado **estático**. Cuando se organizan en enjambres para migrar, estos son numerosos y vuelan en una sola dirección. Estos enjambres en contraposición a los anteriores son llamados **dinámicos**.
- Es obvia el paralelismo que se da entre las principales fases de la búsqueda optimizatoria ([[exploration&exploitation|exploración y explotación]]). La fase dinámica es la paralela a la exploración, ya que se buscan pequeños objetivos en grupos por el espacio. La fase estática (explotación) se da cuando todos los agentes empiezan a migrar a una dirección concreta.

## Algoritmo
### Separación
- $S_i=-\sum_{j=1}^NX-X_j$
- Se refiere a la eludición estática de colisión entre individuos con su vecindario.
- Donde $X$ es la posición del agente actual y $X_j$ la posición de su vecino número $j$. Mientras que $N$ es el número total de agentes.

### Alineamiento
- $A_i=\frac{\sum_{j=1}^N V_i}{N}$
- Se refiere a la coordinación de velocidad dentre individuos de un vecindario.
- Donde $V_j$ es la velocidad del inviduo número $j$ en el vecindario.

### Cohesión
- $C_i=\frac{\sum_{j=1}^N X_j}{N}-X$
- Se refiere a la fuerza de atracción de los individuos hacia el centro de masa del vecindario.

- Se da por hecho un radio alrededor de cada libélula agrupando su vecindario, pues este es muy importante para el comportamiento de esta.
- En un enjambre dinámico, las libélulas tienden a alinear su vuelo al tiempo que mantienen una separación y cohesión adecuadas. En un enjambre estático, sin embargo, las alineaciones son muy bajas mientras que la cohesión es alta para atacar a las presas. Por lo tanto, asignamos a las libélulas pesos de alta alineación y baja cohesión cuando exploran el espacio de búsqueda y de baja alineación y alta cohesión cuando explotan el espacio de búsqueda. Para la transición entre exploración y explotación, los radios de vecindad se incrementan proporcionalmente al número de iteraciones.

![[dragonfly-neighbourds.png]]
### Atracción comida
- $F_i=X^+-X$ 
- La fuerza de atracción hacia la comida (la mejor posición encontrada hasta el momento).
- $X^+$ es la mejor posición encontrada hasta el momento.

### Repulsión depredador
- $E_i= X⁻+X$
- La fuerza de repulsión hacia un depredador (la peor posición encontrada hasta el momento).
- $X^-$ es la peor posición encontrada hasta el momento.

### Dirección
- $\Delta X_{t+1}=(sS_i+aA_i+cC_i+fF_i+eE_i) + w\Delta X_t$
- El delta de $X$ indica el [[vector]] dirección del movimiento de las libélulas. Cada elemento en minúscula es el factor del operador al que multiplica, de forma que es posible acentuar el efecto de cada uno de estos con múltiples combinaciones.

### Posición
- $X_{t+1}=X_{t}+\Delta X_{t+1}$
- El operador de actualización de posición.

