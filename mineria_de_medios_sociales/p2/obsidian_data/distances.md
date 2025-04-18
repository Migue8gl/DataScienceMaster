-  #distances | #clustering | #ml

# Distancia Euclidea
- La distancia euclidea es una de las métricas más comunes, que se define como la raíz cuadrada de la suma de los cuadrados de las diferencias entre las coordenadas de dos puntos.
$$d(x,y) = \sqrt{\sum_{k=1}^n (x_k - y_k)^2}$$

# Distancia de Minkowski
- La distancia de *Minkowski* es una generalización de la distancia euclidea, donde se puede ajustar el parámetro $p$ para obtener diferentes métricas.
$$d(x,y) = \left( \sum_{k=1}^n |x_k - y_k|^p \right)^\frac{1}{p}$$

# Distancia de Mahalanobis
- La distancia de *Mahalanobis* es una métrica que tiene en cuenta la covarianza de las variables, lo que la hace útil cuando las variables tienen diferentes escalas o están correlacionadas. Mide la distancia de un punto al centroide de una distribución normal.
$$d(x,y) = \sqrt{(x-y)\sigma^{-1}(x-y)^T}$$

# Distancia coseno
- La distancia coseno mide la similitud entre dos [[vector|vectores]] a través del coseno del ángulo que forman, y es útil cuando se quiere medir la similitud de patrones independientemente de la magnitud.
$$cos(x,y) = \frac{(x \cdot y)}{||x||||y||}$$

# Correlación de Pearson
- La correlación de *Pearson* es una medida de la relación lineal entre dos variables, y toma valores entre $-1$ y $1$, donde $1$ indica correlación positiva perfecta, $-1$ correlación negativa perfecta, y $0$ ausencia de correlación.
$$\rho_{X,Y} = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$$

# Distancia de Jaccard
- La distancia de *Jaccard* mide la disimilitud entre dos conjuntos, y se define como $1$ menos la razón entre la intersección y la unión de los conjuntos.
$$d_J(A,B) = 1 - J(A,B) = \frac{|A \cup B| - |A \cap B|}{|A \cup B|}$$

# Distancia de Kullback-Leiber
- Es una medida que cuantifica la diferencia entre dos distribuciones de probabilidad. No es una distancia en el sentido matemático estricto, ya que no es simétrica y no satisface la desigualdad triangular, pero es ampliamente utilizada para comparar distribuciones en teoría de la información y aprendizaje automático.
- La divergencia **KL** entre la distribución de probabilidad $P$ y $Q$ en el espacio $X$ es:
$$\sum_{x\in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$