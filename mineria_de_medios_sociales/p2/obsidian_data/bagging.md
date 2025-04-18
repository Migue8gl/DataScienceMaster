- #ensemble | #bagging | #ml

# Concepto
- *Bootstrap aggregation* o *Bagging* es un proceso de propósito general que reduce la varianza de un método de aprendizaje estadístico.
- Dado un conjunto de observaciones $x_{1}, x_{2},...,x_{n}$ la [[variance and standard deviation|varianza]] la media de $X$ es el $\large{\frac{\sigma^2}{n}}$. Por ello, promediando el conjunto de observaciones se consigue reducir la varianza.
- **RandomForest** es un ejemplo de *bagging* con [[decision trees|árboles de decisión]], aunque con una pequeña diferencia (se escogen subconjuntos de variables a la vez que subconjuntos de datos).
- Es altamente paralelizable.
## Bootstrap 
- Para realizar *bagging*, primero se generan $N$ diferentes subconjuntos de datos a partir del conjunto de datos de entrenamiento original. Cada subconjunto es creado utilizando el método de _bootstrap_, que implica tomar muestras al azar con reemplazo (esto significa que una misma observación puede aparecer más de una vez en un subconjunto).
- A continuación, se entrena un modelo (por ejemplo, un [[decision trees|árbol de decisión]]) en cada uno de estos $N$ subconjuntos de datos (conjunto _bootstrappeado_). Esto genera $N$ modelos diferentes, cada uno de los cuales ha sido entrenado en un subconjunto diferente de los datos de entrenamiento.
$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(x)$$

# Out-of-bag Error (OOB)
- #TODO