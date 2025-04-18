- #bayesiannetwork | #modelization

# Relaciones no dirigidas
- Puede ocurrir que existan varias relaciones entre $A,B,C$ en un modelo, pero no es posible asignar dirección a las relaciones. Esta dificultad puede resolverse usando una variable mediadora $D$.
![[bayesian networks construction-var-mediadora.png|400]]
- $D$ posee dos valores: $0,1$. Esto representa el valor de las configuraciones de $A,B,C$.
	- $P(D=1|A,B,C) = R(A,B,C)$
	- $P(D=0|A,B,C)=1-R(A,B,C)$
- $D=1$ significa que determinada configuración entre las variables ocurre.

# Noisy-OR
- Imaginemos una variable $A$ con varios padres. Entonces hay que especificar una distribución para cada configuración $c$ de los padres $P(A|c)$.
- Podría ocurrir:
	- Que algunas configuraciones sean difíciles de considerar o bien haya pocos datos para obtener las probabilidades asociadas.
	- Puede que sea fácil asignar $P(A|X_{i})$ (para algunos padres), pero difícill de asignar la relación conjunta $P(A|X_{1}...X_{n})$.
- **Noisy-OR** es un técnica que:
	- Sean $X_{1}...X_{n}$ las variables binarias que indican las causas que pueden producir un determinado efecto $A$.
	- Cada evento $X_{i}=1$ es susceptible de causar por sí mismo el efecto, a menos que un inhibidor lo impida, con probabilidad $q_{i}$.
	![[bayesian networks construction-modelization-technique.png|500]]
	- Se asume que todos los inhibidores actúan de forma **independiente**:
$$\large{P(A=0|X_{1}=1,...,X_{j}=1)=\prod_{i\in Y}q_i}$$
- Se puede añadir el efecto de un probabilidad residual $\lambda_0$ que haga efecto sin tener en cuenta ninguna variable, es una modelización del ruido.
 - Ejemplo:
![[bayesian networks construction-noisy-or.png]]

# Divorcio de padres
- Sea $X_{1},...,X_{n}$ una lista de variables que causan el efecto $A$. Se desea asignar probabilidades a la distribución $P(A|X_{1},...,X_{n})$, lo que podría ser muy difícil de conseguir. Para manejar esta situación suele aplicarse una técnica conocida como **divorcio de padres**, introduciendo variables mediadoras y reduciendo el conjunto de padres de la variable $A$.
![[bayesian networks modelization techniques-parent-separation.png|500]]

# Redes bayesianas orientadas a objetos
- En redes complejas suele haber varias copias de fragmentos casi idénticos que se repiten varias veces. Supongamos el modelo siguiente:
![[bayesian networks modelization techniques-object-oriented.png|500]]
- Se asume que:
	- $X_{1}$ y $X_{2}$ tienen el mismo conjunto de estados que la distribución de probabilidad de las variables $A_{1},A_{2},A_{3}$ y $A_{4}$ es idéntica.
	- Lo mismo ocurre con $B_{i}, C_{i}, D_{i},E_{i}$.
- Asumiendo esas condiciones, se observa que la red tiene $4$ copias del misma fragmento, que contiene las variables $A_{i},B_{i},C_{i},D_{i}$ y $E_{i}$. Esto puede aprovecharse, creando una plantilla genérica para el fragmento repetido, que se denomina **clase**, siguiendo la terminología propia de orientación a objetos. Cada uso del patrón se denominará **objeto** o **instancia**.
![[bayesian networks modelization techniques-object-oriented-2.png|550]]
- Para poder especificar la distribución de probabilidad de la variable $A$ se incluye el nodo artificial $X$, con el mismo conjunto de estados que $X_{1}, X_{2}$.
- Los nodos $D,E$ se corresponden con la parte del objeto que es accesible desde el exterior, ya que pueden ser padres de variables fuera del objeto.
- Los nodos $A,B,C$ son transparentes para el resto del modelo.
![[bayesian networks modelization techniques-oo-ex3.png|500]]