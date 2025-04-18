- #semisupervisedlearning | #ml

# Concepto
- Es un tipo de aprendizaje que combina el aprendizaje supervisado con el aprendizaje supervisado.
- Se tienen una serie de datos etiquetados (no muchos) y una gran mayoría de datos no etiquetados. También puede darse el caso contrario en el que hay muchos datos etiquetados y unos pocos no etiquetados.
- La idea principal es tratar de mejorar la información que aportan los datos no etiquetados para enriquecer los datos etiquetados, aprovechando ambos tipos de información para enriquecer el aprendizaje.
![[semi supervised learning-example.png]]

# S3VMs
- Son los algoritmos de [[support vector machine]] adaptados al semi supervisado. Se impone la restricción de que los datos no etiquetados deben estar en regiones con menor densidad de datos.
![[semi supervised learning-s3vm.png]]

# Algoritmos basados en grafos
- Supongase que ejemplos de datos muy similares probablemente tienen la misma etiqueta. Si se tienen muchos datos etiquetados, esto sugiere un algoritmo del tipo [[k-nearest neighbors]]. Si se tienen muchos datos no etiquetados entonces se pueden usar como pequeños "peldaños".
- La idea principal es construir un grafo con aristas entre ejemplos muy similares. Los datos no etiquetados pueden ayudar a agrupar observaciones de la misma clase.
- Para buscar el límite de clasificación se usan los ejemplos etiquetados, particionando el grafo en subgrafos (cortes mínimos) -> Algoritmo heurístico de optimización.
![[semi supervised learning-graph-based.png]]

# Label propagation
- Se conectan las observaciones que están cerca unas de otras y se propaga la clase de las observaciones etiquetadas a las no etiquetadas.
![[semi supervised learning-label-propagation.png|300]]
## k-Nearest Neighbor Graph (kNNG)
- Se añaden ejes entre la instancia y sus $k$ vecinos más cercanos.
- Crean grafos asimétricos (quizá el vecino más cercano de $a$ es $b$, pero no al revés). Además son poco escalables.
## e-Neighborhood
- Se añaden ejes a todas las instancias dentro de un radio $e$ con respecto a las instancias etiquetadas.
- No es escalable y es muy sensible al parámetro $e$. Puede crear grafos con componentes desconectadas.

# Menciones
- *Co-Training Algorithm*, *Tri-Training Algorithm*, *Multi View*, *Single View*, *NCA*, *Metric Learning*.
- *Multiple-classifier* + *single view* es lo que parece ir mejor.