- #pam | #clustering | #kmedoids

# Concepto
- Es un algoritmo de *clustering* muy similar a [[clustering|k-means]].
- En lugar de calcular el centroide como una [[mean and weighted mean|media]] (lo cual podría representarse en un punto no existente), se calcula un ejemplo real com tal.
- $K$-Medoides minimiza la distancia entre puntos del clúster y los puntos considerados como centroides.
- Aumenta la interpretabilidad
![[k-medoids-example.png]]

# Ventajas y desventajas
- Menos sensible a ruido que el $k$ medias, pues los [[outliers]] no pueden afectar tanto al centroide ya que es un punto real en el espacio. En el caso de $k$ medias los *outliers* son capaces de afectar mucho más una agrupación dada debido a que el centroide de esa agrupación se calcula como una media y no está restringido a existir, por ello puede desplazarse más.
- $K$-Medoides es mucho más lento que $k$-medias, ya que necesita calcular la distancia de todos los puntos para obtener el medoide, mientras que en $k$ medias, al tratarse de una media, es simplemente una operación.