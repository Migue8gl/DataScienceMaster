- #hierarchicalclustering | #graphs

# Bottom Up
- El *clustering* aglomerativo comienza con todos los nodos desconectados, considerado cada uno de ellos como una comunidad independiente.
- Las similitudes entre pares de nodos $(i,j)$ se calculan como un peso $w_{ij}$ usando distintas medidas de [[distances|distancias]].
- La similitud debe ser alta para pares de nodos que tengan una probabilidad alta de pertenecer a la misma comunidad y baja para los que probablemente sean de comunidades distintas.
- Se van enlazando grupos por pares, mezclando comunidades en cada paso, de forma que se van creando *clusters* de forma creciente y jerárquica.
- Se obtiene finalmente un dendograma del que se pueden analizar distintas particiones cortando por el nivel deseado.
![[hierarchical clustering-dendogram.png]]