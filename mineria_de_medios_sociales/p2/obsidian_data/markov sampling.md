- #bayesiannetwork | #sampling | #montecarlo | #markovsampling

# Funcionamiento
- Se les asigna a las variable evidencia su correspondiente valor.
- Se simula la red de manera estocástica:
	- Se genera una realización aleatoria, eligiendo al azar o bien aplicando el método de [[logic sampling]] o [[likelihood weighting]].
	- Se simulan las variables no evidenciales, una a una, siguiendo un orden arbitrario, mediante su función de probabilidad condicionada a todas las demás $p(x_{i}|x\backslash x_{i})$.
- El peso asociado a cada realización es siempre igual a $1$.
- La función de probabilidad condicional $p(x_{i}|e)$ se estima por la proporción de realizaciones en las que ocurre $x_{i}$.
## Función de una variable condicionada a todas las demás
$$h(x_{i})=p(x_{i}|x\backslash x_{i})\propto p(x_{i}|\pi_{i})\prod_{X_{i}\in C_{i}}p(x_{j}|\pi_{j})$$
- Esto significa que la probabilidad condicional $p(x_{i}|x\backslash x_{i})$ **es proporcional** al producto del término $p(x_{i}|\pi_{i})$ y las probabilidades condicionales de sus hijos en $C_{i}$.
- Una vez muestreadas todas las variables no evidenciales, se obtiene una realización. Los valores obtenidos se utilizan para generar la siguiente realización.

# Ejemplo
![[markov sampling-ej1.png]]
![[markov sampling-ej2.png]]
![[markov sampling-ej3.png]]
![[markov sampling-ej4.png]]
![[markov sampling-ej5.png]]
![[markov sampling-ej6.png]]
![[markov sampling-ej7.png]]