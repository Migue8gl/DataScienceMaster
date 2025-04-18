- #silhouette | #clustering

# Concepto
- Cuantifica cómo de lejano es un objeto de un clúster a los objetos de los otros clústeres; esto es, si está separado del borde o no.
- Para cada objeto, se suma por un lado la distancia con todos los objetos de su clúster (a) y por otro, la distancia con los objetos del clúster más cercano (b). Finalmente, se calcula el ratio entre ambas sumas.
- En $[-1, 1]$, mejor cuanto más próximo a $1$.
$$a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, j \neq i} d(i,j)$$
$$b(i) = \min\limits_{k\neq i} \frac{1}{|C_k|} \sum_{j \in C_k} d(i,j)$$
$$s(i) = \begin{cases}
  1 - a(i)/b(i), & \text{if } a(i) < b(i) \\
  0, & \text{if } a(i) = b(i) \\
  b(i)/a(i) - 1, & \text{if } a(i) > b(i)
\end{cases}$$

![[silhouette-example.png]]