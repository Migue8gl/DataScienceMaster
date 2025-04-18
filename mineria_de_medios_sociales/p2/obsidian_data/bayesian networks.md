- #bayes | #ml | #statistics 
- [[mgp exercicies|ejercicios]]

# Concepto
- Una red bayesiana o red de creencia/decisión es un modelo gráfico probabilístico que representa un conjunto de variables y sus dependencias condicionales por medio de un [[graph theory|grafo acíclico]] (**DAG**).
- Las redes bayesianas son ideales para, por medio de un evento que ha ocurrido, predecir la probabilidad de contribución de cada una de las posibles causas. Un ejemplo es las relaciones probabilísticas entre enfermedad y síntomas.
- Se representan como un conjunto de variables y un conjunto de enlaces dirigidos entre variables. Cada variable tiene un conjunto de estados finito y exclusivo. A cada variable $A$ con padres $B_{1},..., B_{n}$ una tabla con los valores de la distribución condicional $P(A|B_{1}...B_{n})$.
![[bayesian networks-example-assigned-probabilities.png|500]]
- Se puede generar la distribución conjunta de la red bayesiana por medio de las distribuciones individuales (y gracias a la [[d-separation|d-separación]] no es necesario usar todas las distribuciones).
## Orden ancestral
- En **redes bayesianas**, un **orden ancestral** es una forma de organizar los nodos de acuerdo con su relación de dependencia causal, es decir, en un orden tal que **cada nodo aparece después de sus padres en la red**.

# Inserción de evidencia
- Muchas veces interesa conocer la forma en que disponer de información (**evidencia**) modifica la creencia sobre la probabilidad de ocurrencia de las variables de la red. Esta evidencia se representa como $e$ para indicar que es un vector.
- Se asume que la evidencia sobre una variable $X$ con $n$ estados se representa con una tabla llena de ceros y unos. Los ceros corresponden con los valores no observados. A esto se le conoce como potencial $\phi_{e}(X)$.
- Imaginemos la distribución $P(A|B,C)$ donde todas las variables tienen dos estados posibles:
$$
\begin{array}{c|cccc}
BC & b_1c_1 & b_1c_2 & b_2c_1 & b_2c_2 \\
\hline
A = a_1 & 0.3 & 0.6 & 0.9 & 0.8 \\
A = a_2 & 0.7 & 0.4 & 0.1 & 0.2
\end{array}
$$
- Y queremos incorporar la evidencia $A=a_{1}$. Para incorporar la evidencia se usa un potencial para representarla, que llamaremos $\phi_{A=a_{1}}(A,B,C)$.
$$
\begin{array}{c|cccc}
BC & b_1c_1 & b_1c_2 & b_2c_1 & b_2c_2 \\
\hline
A = a_1 & 1 & 1 & 1 & 1 \\
A = a_2 & 0 & 0 & 0 & 0
\end{array}
$$
- Solo hay valores de $1$ asociados a las configuraciones de $A=a_{1}$.
- Incorporar la evidencia implica hacer la multiplicación:
$$P(A|B,C)\phi_{A=a_{1}}(A,B,C)$$
- Esta multiplicación produce como resultado un potencial que solo depende de $B$ y de $C$, que denominaremos $\phi_{2}(B,C)$ y que se representa abajo:
$$
\begin{array}{c|cccc}
BC & b_1c_1 & b_1c_2 & b_2c_1 & b_2c_2 \\
\hline
& 0.3 & 0.6 & 0.9 & 0.8 
\end{array}
$$
![[bayesian networks-example-evidence.png|500]]

# Construcción
- [[bayesian networks construction]]
- [[structure learning]]

# Problema fundamental
- Tenemos una red bayesiana asociado a un conjunto de probabilidades. El **problema fundametal** es dado un conjunto $O$ de variables observadas y una variable objetivo $Z$, queremos calcular $P(Z|O)$ para todos lo valores de $Z$.
- Calcular la probabilidad condicionada tiene una complejidad **exponencial** en el número de variables.

# Reglas de cálculo (Inferencia exacta)
- En estas reglas de cálculo se basan las redes bayesianas para el cómputo exacto de la probabilidad.
## Marginalización
$$P(X)=P(X,Y=y_{1}) + P(X,Y=y_{2})+...+P(X,Y=y_{m})=\sum_{i}^{m}P(X,Y=y_{i})$$
## Cálculo de la distribucion condicional
$$P(X|Y)=\frac{P(X,Y)}{P(Y)}$$
$$P(Y|X)=\frac{P(X,Y)}{P(X)}$$
## Combinación marginal y condicional
$$P(X,Y)=P(X|Y)\cdot P(Y)=P(Y|X)\cdot P(X)$$
## Regla de la cadena
$$P(X_{1},X_{2},...,X_{n})=\prod_{i}P(X_{i}|X_{1},...,X_{n-1})$$
![[bayesian networks-conditional-probability-table.png]]

# Algoritmos de aproximación
- **Métodos de Monte Carlo**: Generan una muestra $\{x^{j}\},j=1,...,m$ de configuraciones de las $n$ variables a partir de la distribución conjunta de la red y luego aproximan las probabilidades de cada caso de la variable, $X_{k}=x_{k}$, como la frecuencia relativa de dicho caso en la muestra. Algunos ejemplos son los algoritmos de [[logic sampling|muestreo lógico]], [[likelihood weighting|ponderación por verosimilitud]] o [[markov sampling|muestreo de Markov.]]
- **Métodos deterministas**: Obtienen una aproxiación para $p(X_{k}|e)$ que es siempre a misma en cualquier ejecución del algoritmo si se usan los mismos parámetros de entrada.
## Precisión de la aproximación
- Se suele calcular el error de la aproximación con el [[mean squared error|RMSE.]]
- Dados $p(x_{i}|e)$ y su aproximación $\hat{p}(x_{i}|e)$ tiene la distancia de [[distances|*Kullback-Leiber*]] para $p(x_i|e)$:
$$\sum_{x_i \in \varOmega_{X_i}} p(x_i|e) \log \left( \frac{p(x_i|e)}{\hat{p}(x_i|e)} \right)$$