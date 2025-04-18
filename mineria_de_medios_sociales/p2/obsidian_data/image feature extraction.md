- #datascience | #computervision | #featureextraction

# Descriptores
- En extracción de características de imágenes, los _descriptores_ son representaciones matemáticas o [[vector|vectores]] que capturan información específica de una imagen o de una región de la imagen para poder analizarla, clasificarla o reconocerla. Los descriptores permiten transformar la información visual (píxeles) en una representación numérica, facilitando así su procesamiento.
- A la hora de calcular un **descriptor**, ya sea **LBP** u otro, es frecuente analizar la imagen en bloques (en lugar de un único cálculo global).
- Se divide la imagen en bloques de igual tamaño y se calculan histogramas por cada bloque. El vector de rasgos resultante es la concatenación en un orden preestablecido de los histogramas de cada bloque.
- Cada histograma se suele normalizar dividiéndolo por $||h||_{2}$. De esta forma se obtiene que -> $\sum_{i}h^{2}_{i}=1$.
- En la práctica se utilizan bloques solapados para ganar robustez.
## Tipos
- [[texture]]
- [[local binary patterns]]