- #ordinal | #monotonic | #ml

# Concepto
- También llamada *ranking/sorting*, es un problema de aprendizaje supervisado para predecir categorías que tienen una disposición ordenada.
![[ordinal and monotonic learning.png]]
- Los costes en aprendizaje ordinal no son los mismos que un aprendizaje normal.

# Conversión a ordinal
## Regresión
- En problemas de clasificación ordinal, las clases tienen un orden (por ejemplo, "bajo", "medio", "alto"), pero no necesariamente una distancia numérica definida entre ellas.
- Para aplicar técnicas de regresión, cada categoría se transforma en un valor numérico, por ejemplo, asignando $1$ a "bajo", $2$ a "medio" y $3$ a "alto".
- Se entrena una función de regresión en estos valores numéricos.
- Dado que la regresión produce valores continuos es necesario redondearlos para obtener una categoría discreta. Pese a ello, la elección de una función de mapeo noes trivial, pues las diferencias entre categorías no siempre son iguales.
## Matrices de costo
- Se puede definir una matriz de costo para diferenciar los distintos tipos de costo para cada clase en función a las otras clases.
## Multiple Model for Ordinal Classification
- Si hay $k$ clases, se generan $k-1$ modelos binarios.
- Cada modelo responde a la pregunta: ¿La instancia pertenece a una clase mayor o igual a cierto umbral? Esto significa que cada modelo aprende a distinguir entre clases inferiores y superiores en la jerarquía ordinal.
- Para una nueva instancia, cada modelo binario devuelve una probabilidad y se combinan los resultados para determinar la clase ordinal final, generalmente usando un umbral de decisión.

# Aprendizaje monotónico
- Es un tipo de clasificación ordinal donde una restricción monotónica es clara: un valor más alto de un atributo en un ejemplo, fijando el resto de valores, no debería reducir el valor de la asignación de clase. Por ejemplo, si los años de experiencia aumentan, el salario siempre debe ser igual o mayor.
$$x_{1}\ge x_{2}\implies f(x_{1})\ge f(x_{2})$$
- Se se cumple alguna de las restricciones lógicas, entonces es no monotónica:
$$
X < Y \wedge C_x > C_y  
$$
$$
X > Y \wedge C_x < C_y  
$$
$$
X = Y \wedge C_x \neq C_y  
$$
- Para aplicar monotonicidad a vectores, el orden natural debe extenderse para permitir órdenes de vectores en $\mathbb{R}^{p}$. A estas relaciones se les conoce como relaciones parciales.
$$
x \preceq x' \iff \forall i = 1..p, \quad x_i \leq x_i'
$$
- Función monotónica completa:
$$
x \preceq x' \Rightarrow f(x) \leq f(x'), \quad \forall x, x' \in \mathbb{R}^p
$$
- Las funciones parcialmente monótonas pueden tener atributos monótonos y otros no monótonos. El espacio de entrada se puede dividir en dos conjuntos, uno de atributos monótonos y otros de los que no. De esta forma, la función monótona se define como:
$$x \preceq x' \Rightarrow f(x,z) \leq f(x',z), \quad \forall x, x' \in X,z\in Z$$

![[ordinal and monotonic learning-taxonomy.png]]