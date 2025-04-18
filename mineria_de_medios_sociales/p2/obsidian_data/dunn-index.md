- #dunnindex | #clustering

# Concepto
- Cuantifica el ratio entre la distancia mínima entre puntos de diferentes clústeres y el diámetro máximo de los clústeres.
- Cuanto más alto mejor.
$$\large{Dunn=min_{1\leq i\leq k}(min_{k}(\frac{\delta(c_{i},c_{j})}{max_{1\leq i\ne i\leq k}(\Delta(C_k))}))}$$
- Donde $\delta(c_i,c_i)$ es la distancia entre todos los pares de clústeres $i,j$.
- Donde $\Delta(C_{k})$ es la distancia máxima entre objetos del clúster $k$.
![[dunn-index-example.png|520]]
