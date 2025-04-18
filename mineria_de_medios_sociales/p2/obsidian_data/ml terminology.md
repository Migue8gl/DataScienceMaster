- #ml | #notation

- **Samples** : instances, observations.
	- Si los datos son tabulares, son las filas de la tabla. Cada elemento medido que se compone de una serie de características y proviene de una población.
- **Features**: inputs, attributes, measurements, dimensions.
	- Las características medidas, las columnas de la tabla. Los datos que dan información acerca de la población.
- **Class labels**: targets, outcomes.
	- En aprendizaje supervisado, las etiquetas o clases a predecir.
- **Loss function**: A veces también llamada *error function*. Es una función de coste a minimizar. El término *loss* suele estar asociado a la pérdida medida en un solo punto y *cost* es una medida que calcula la pérdida de todo el dataset.

- Un conjunto de datos se puede representar como una matriz o una tabla. Matemáticamente como $X\in\mathbb{R}^{n\times m}$, donde $n$ son las filas o intancias y $m$ las columnas o features.
- Se usan letras minúsculas para representar [[vector|vectores]] ($x$) y letras en mayúsculas para matrices ($X$).
- Elementos sueltos en vectores y matrices con super índices, por ejemplo $x_i$ o $x_{ij}$.