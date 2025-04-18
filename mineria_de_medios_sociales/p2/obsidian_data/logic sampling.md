- #logicsampling | #bayesiannetwork | #sampling

# Concepto 
- Este algoritmo está basado en el teorema de aceptación-rechazo. Este es un método que se usa para generar muestras de una distribución objetivo $f(x)$, cuando es difícil muestrearla directamente. Se usa una distribución propuesta $g(x)$, de la cuál es más fácil generadas muestras, y en un factor de escalado $M$ tal que:
$$f(x)\leq Mg(x), \quad \forall x$$

# Funcionamiento
- Se genera una muestra $X$ de la distribución $g(x)$.
- Se genera un número aleatorio $U\textasciitilde U(0,1)$.
- Se acepta $X$ si $U\leq \frac{f(X)}{Mg(X)}$ y se rechaza en caso contrario.
- Se repite hasta generar tantas muestras como se desee.
- Es uno de los métodos de propagación hacia adelante. Se muestrea una variable solo cuando ya han sido muestreados todos sus padres.
- Se simulan todas las variables, incluso las evidenciales.
- La función de simulación usada para $X_{i}$ es su función de probabilidad condicional:
$$h(x_{i}|\pi_{i})=p(x_{i}|\pi_{i}), \quad i\in\{1,...,n\}$$
- Los pesos se obtienen con:
$$s(x)=\frac{p_{e}(x)}{h(x)}=\frac{\prod_{x_{i}\notin E}p_{e}(x_{i}|\pi_{i})\prod_{x_{i}\in E}p_{e}(x_{i}|\pi_{i})}{\prod_{x_{i}\notin E}p(x_{i}|\pi_{i})\prod_{x_{i}\in E}p(x_{i}|\pi_{i})}$$
- Lo que es igual a:
$$s(x)=\begin{cases} 1, & \text{si } x_i = e_i, \; \forall X_i \in \mathbf{E} \\ 0, & \text{en otro caso} \end{cases}$$
- Si $x_{i}\ne e_{i}$ para algún $X_{i}\in E$, entonces el peso es cero y se **rechaza la muestra**.
![[logic sampling-ej1.png]]
![[logic sampling-ej2.png]]
![[logic sampling-ej3.png]]
![[logic sampling-ej4.png]]
![[logic sampling-ej5.png]]