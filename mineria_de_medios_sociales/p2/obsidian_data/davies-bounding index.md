- #metrics | #clustering

# Concepto
- El índice de **Davies-Boulding** cuantifica la "separabilidad media" de un clúster frente al clúster más cercano. Esto significa que mide qué tan bien separados están los diferentes clústeres entre sí.
- Para calcularlo, se usa la varianza dentro de los clústeres (dispersión o scatter) y la separación entre los centroides (centros) de los clústeres. Cuanto menor sea la varianza dentro de los clústeres y mayor la separación entre sus centroides, mejor será la separación de los clústeres.
- La fórmula para calcular el índice de Davies-Boulding es:
$$DB = \frac{1}{N} \sum_{k=1}^N R_k$$
- Donde $R_k$ es el máximo ratio entre la suma de la dispersión dentro de cada clúster y la distancia entre sus centroides. $N$ es el número total de clústeres.
![[davies-bounding index-example.png|400]]