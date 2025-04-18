- #mh | #wao | #whaleoptimization

## Inspiración
- Simula el comportamiento de las ballenas jorobadas en el comportamiento de caza simulado con el azar o el mejor agente de búsqueda para perseguir a la presa y el uso de una espiral para simular el mecanismo de ataque de la red de burbujas de las ballenas jorobadas.

![[whale-optimization-algorithm.png|500]]

## Algoritmo
### Rodeando a la presa
- $\vec{D}=|\vec{C}.\vec{X}^*(t)-\vec{X}(t)|$
- $\vec{X}(t+1)=\vec{X}^*(t)-\vec{A}·\vec{D}$
- Donde $t$ indica la iteración actual, $\vec{A}$ y  $\vec{C}$ son vectores de coeficientes, $X^*$ es el vector de posición de la mejor solución encontrada hasta el momento y $\vec{X}$ es el vector posición.

 - Los vectores $\vec{A}$ y $\vec{C}$ se calculan como sigue:
	 - $\vec{A}=2\vec{a}·\vec{r}-\vec{a}$
	 - $\vec{C}=2·\vec{r}$
- Donde $\vec{a}$ es decrementado linealmente desde $2$ hasta $0$ y $\vec{r}$ es in vector aleatorio con valores en $[0,1]$.

### Ataque de red de burbujas (explotación)
#### Mecanismo de encogimiento del cerco
- Este mecanismo es conseguido mediante el decremento del valor de $\vec{a}$. El vector $\vec{A}$ es un vector de valores aleatorios en el intervalo $[-a,a]$ donde $a$ es decrementado desde $2$ hasta $0$.

![[wao-shrinking-circle-prey-mechanism.png|350]]

#### Mecanismo de actualización de posición en espiral
- Este mecanismo calcula la distancia entre la ballena $(x,y)$ y la presa $(x^*, y^*)$. Una ecuación con forma de espiral es creada entonces entre la ballena y la presa para simular el comportamiento de estas al cazar.
- $\vec{X}(t+1)=\vec{D}^{'}·e^{bl}·cos(2\pi l)+\vec{X}^*(t)$
- Donde $\vec{D}^{'}=|\vec{X}^*(t)-\vec{X}(t)|$ e indica la distancia entre presa y ballena. $b$ es una constante para definir la forma de la espiral, $l$ es un número aleatorio entre $[-1,1]$.

![[spiral-update-position-wao.png]]

- Se utilizan ambos mecanismos, dependiendo de si cierto número $p$ aleatorio es mayor o menor a $0.5$.

### Búsqueda de la presa (exploración)
- El mismo método utilizando el vector $\vec{A}$ puede ser utilizado para la exploración. Se puede usar $\vec{A}$ con valores más grandes a $1$ o menores a $-1$ para forzar al agente a moverse lejos de otro agente.
- $\vec{D}=|\vec{C}·\vec{X}_{rand}-\vec{X}|$
- $\vec{X}(t+1)=\vec{X}_{rand}-\vec{A}.\vec{D}$
- Cuando $|A|\geq 1$ -> [[exploration&exploitation|Exploración]], cuando $|A|<1$ -> explotación.