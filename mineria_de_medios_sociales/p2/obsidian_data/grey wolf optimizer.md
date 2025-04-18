#gwo | #mh 

## Inspiración
- El algoritmo se inspira en el comportamiento de la manada de los lobos grises y sus escala jerárquica. Los lobos grises son superdepredadores, es decir, están en la cima de la cadena alimentaria. Su escala jerárquica es la siguiente:
	- $\alpha$ -> Los lobos alfa son aquellos que lideran la manada, no por su fuerza, si no por su capacidad organizativa dentro de la manada. Estos solo puede procrear dentro de la manada y son los encargados de tomar decisiones importantes sobre todo el conjunto.
	- $\beta$ -> Los lobos beta son subordinados directos de los alfa. Son la opción más directa cuando se trata de sustituir a un alfa. Deben mostrar respeto a los alfa y comandan al resto de la manada según dicte el alfa.
	- $\omega$ -> El nivel más bajo de la cadena. Son los últimos en comer, pero no por ello carecen de importancia, pues la manada puede llegar a enfrentarse a graves problemas cuando este nivel de lobos falta. Son totalmente necesarios para mantener la estructura.
	- $\delta$ -> Lobos que someten a los omegas, pero que debem someterse a los alfas y omegas. A este grupo pertenecen los cazadores, ancianos, centinelas y cuidadores.

## Modelado de jerarquía
- Se considera la solución más buena como la solución $\alpha$, siendo las siguientes en orden de mejor a peor [[fitness function|fitness]] la $\beta$, la $\delta$ y por último la $\omega$.
- La caza (optimización) es guiada por las soluciones $\alpha$, $\beta$ y $\delta$ mientras que las soluciones $\omega$ siguen a estos tres lobos.

## Operadores

### Rodeo de la presa
- $\vec{D}=|\vec{C}\cdot\vec{X}_p(t)-\vec{X}(t)|$
- $\vec{X}(t+1)=\vec{X}_p(t)-\vec{A}\cdot\vec{D}$
- Donde $t$ indica la iteración actual, $\vec{A}$ y $\vec{C}$ son vectores de coeficientes, $\vec{X}_p$ es el vector posición de la presa y $\vec{X}$ el vector posición del lobo gris.
- Los vectores $\vec{A}$ y $\vec{C}$ se calculan:
	- $\vec{A}=2\vec{a}\cdot\vec{r}_1-\vec{a}$
	- $\vec{C}=2\cdot\vec{r}_2$
- Donde los componentes del vector $\vec{a}$ son decrementados de $2$ a $0$ linealmente según avanzan las iteraciones. Las variables  $r_1$ y $r_2$ son vectores aleatorios entre $[0,1]$. 

### Caza
- Se supone que el alfa, beta y delta tienen mayor conocimiento de donde está la presa (óptimo) y por tanto la posición del resto de lobos debe actualizarse siguiendo a estos tres primeros.
	- $\vec{D}_{\alpha}=|\vec{C}_1\cdot\vec{X}_{\alpha}-\vec{X}|,\quad \vec{D}_{\beta}=|\vec{C}_2\cdot\vec{X}_{\beta}-\vec{X}|,\quad \vec{D}_{\delta}=|\vec{C}_3\cdot\vec{X}_{\delta}-\vec{X}|$
	- $\vec{X}_1=\vec{X}_{\alpha}-\vec{A}_1\cdot\vec{D}_{\alpha},\quad \vec{X}_2=\vec{X}_{\beta}-\vec{A}_2\cdot\vec{D}_{\beta},\quad \vec{X}_3=\vec{X}_{\delta}-\vec{A}_3\cdot\vec{D}_{\delta}$
	- $\vec{X}(t+1)=\frac{\vec{X}_1+\vec{X}_2+\vec{X}_3}{3}$

### Ataque a la presa (explotación)
