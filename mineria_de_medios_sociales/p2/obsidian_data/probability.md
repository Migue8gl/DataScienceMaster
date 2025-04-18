- #probability

# Introducción
- La probabilidad es el estudio teórico de medir la certeza de que un evento ocurra.
- Suele expresarse como un *porcentaje*. De manera más común, esta probabilidad será un número entre $0.0$ y $1.0$.
- Sea la probabilidad del evento $x\rightarrow P(X)$.
- La *verosimilitud* (**likelihood**) es distinta a la *probabilidad* (**probability**). Las diferencias son que la probabilidad mide la frecuencia con que ciertos eventos van a ocurrir. La verosimilitud mide la frecuencia de eventos que ya han ocurrido. 
- **Es una forma de expresar como de probable es que haya ocurrido un evento**, no la probabilidad de que vaya a ocurrir. Los datos son observados y a partir de ahí se pretende estimar el parámetro más probable para que haya ocurrido eso.
- Si la probabilidad de un evento ocurriendo es:
	- $P(X)=0.7$
- La probabilidad de que no ocurra es:
	- $P(not\hspace{2mm}X)=1-0.7=0.3$
- Otra distinción entre likelihood y probabilidad es que las probabilidades de todos los eventos mutuamente excluyentes, es decir, que solo uno de ellos puede ocurrir a la vez, debe sumar $1.0$ o $100\%$ equivalentemente. Likelihood no está sujeta a esta regla.
- Alternativamente, la probabilidad puede ser expresada (**Odds**) como fracciones o número de la siguiente forma (en vez de porcentajes).: ^3291b4
	- $O(X)=\frac{P(x)}{1-P(x)}$ | $P(x)=\frac{O(x)}{1+O(x)}$
- La interpretación dada podría ser, si tenemos $O(X)=2$, el evento es dos veces más probable que ocurra a que no lo haga. 

# Perspectivas
## Frecuentista
- Equipara la probabilidad de una ocurrencia de un evento con la frecuencia relativa de ese evento en un gran número de repeticiones de un experimento.
## Subjetiva
- Interpreta la probabilidad como un grado de creencia personal en la ocurrencia de un evento, basado en información y juicios previos.  

# Matemáticas
- [[joint probabilites]]
- [[union probabilites]]
- [[conditional probability]]
- Si $A\cap B=\varnothing$ entonces $P(A\cup B)=P(A)+P(B)$.
- Si $A\cap B\neq\varnothing$ entonces $P(A\cup B)=P(A)+P(B)-P(A\cap B)$.
## Estados
- Para una variable $A$ con estados $a_{1},...,a_{n}$, expresamos la incertidumbre sobre el estado de la variable como una distribución de probabilidad $P(A)$ sobre sus estados:
	- $P(A)=\{p(a_{1}, ..., p(a_{n}\}, p(a_{i})\geq 0$
	- $\sum_{i}^{n}p(a_{i})=p(a_{i})+...+p(a_{n})=1$
- Usamos la notación $P(A=a_{i})= P(a_{i})$.
### Probabilidad conjunta
- Ocurre cuando se intenta calcular la probabilidad de resultados simultáneos para varias variables, es decir:
	- $P(A,B)=P(A\cap B)$ tiene $m\times n$ posible valores y la suma de la combinación de cada valor de las variables debe ser $1$.
	- $\sum_{i=1}^{n}\sum_{j=1}^{m}P(A=a_{i}, B=b_{j})=1$
### Marginalización
- Dada una distribución conjunta puede calcularse las tablas de las distribuciones **marginales**. Dado $P(A,B)$ se calcula la probabilidad de $P(A)$ considerando que las salidas de $B$ pueden ocurrir de manera conjunta con cada estado $a_{i}$.
- $P(A=a_{i})=\sum_{j=i}^{m}P(A=a_{i}, B= b_{i})$
- $P(A)=\sum_{B}P(A,B)$
- Si se tiene una distribución [[normal distribution|normal]] multivariante, entonces las distribuciones marginales (distribuciones de cada una de sus características/variables) también son normales.
### Independencia condicional
- Se dice que dos variables $A$ y $B$ son condicionalmente independientes dada otra varriable $C$, si:
	- $P(a_{i}|b_{j}, c_{k})=P(a_{i}|c_{k})$
- Esto indica que cuando se observa el valor de $C$ entonces tener información sobre $B$ ya no aporta información adicional sobre el valor de $A$.
- Cuando dos variables son condicionalmente independientes, entonces la **regla fundamental** se simplifica:
	- $P(A,B|C)=P(A|C)P(B|C)$
### Potenciales
- [[potentials]]