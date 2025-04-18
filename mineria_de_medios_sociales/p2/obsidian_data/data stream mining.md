- #dataflow | #datascience | #datamining

# Introducción
- La **minería de flujo de datos** (_data stream mining_) es una rama de la minería de datos que se centra en el procesamiento y análisis de datos que llegan en forma de flujos continuos y en tiempo real. A diferencia de los enfoques tradicionales de minería de datos, que trabajan con conjuntos de datos estáticos y almacenados, la minería de flujo de datos debe lidiar con grandes volúmenes de datos en constante cambio.

# Conceptos
- [[concept drift]]

# Clasificación
- Las primeras aproximaciones se clasificación en un ámbito de constante flujo de datos se basan en algoritmos de aprendizaje incremental o [[online machine learning]].
## Evaluación
- Se busca definir una imagen precisa del rendimiento a lo largo del tiempo. Para ello se utiliza el **Holdout**.
- Se toman instantáneas en diferentes momentos durante el entrenamiento del modelo para ver cómo varía la métrica de calidad. Sólo es válido si el conjunto de *test* es similar a los datos actuales (sin *concept drift*).
- También existe la técnica *interleaved test-then-train* o *prequential*. Esta utiliza cada ejemplo individual para probar el modelo antes de usarlo para entrenarlo. Cuando la evaluación se realiza en este orden, el modelo se está probando en casos que no ha visto y estos datos se aprovechan para un posterior entrenamiento. 
- En *test-then-train* se tienen en cuenta todos los datos vistos hasta la fecha para calcular la métrica de calidad, mientras que en *prequential* solo se consideran los más recientes mediante una ventana deslizante.
![[data stream mining-use-cases.png]]