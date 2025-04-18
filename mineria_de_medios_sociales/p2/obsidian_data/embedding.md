- #dl | #nlp | #ml | #embedding

# Concepto
- Un **embedding** es una representación numérica (normalmente un [[vector]] de números reales) que captura el significado o características esenciales de un dato, como una palabra, oración, imagen o usuario, en un espacio de menor dimensión.

# Tokenization
- El proceso de **tokenizado** consiste en dividir un texto en una serie de partes o *chunks*:
- Al hacerlo, es posible dividir un texto en pequeñas partes que conforman un vocabulario. Cada una de estas partes es codificable y con un significado semántico.
![[tokenization.png|500]]
# Embedding
- Un modelo contiene lo que se define como **vocabulario**, esto es una lista de palabras embebidas que el modelo conoce.
- El **embedding** es el proceso de codificar cada *token*, de forma que pasa a ser un [[vector]] $n$-dimensional.
![[embedding-matrix.png]]
- Los parámetros de cada *token* suelen inicializarse de manera aleatoria inicialmente, aunque más tarde se ajustan los pesos.
- Al codificar las palabras como vectores, se les puede dar un significado semántico a la dirección que estas toman en el espacio multidimensional.
- La dimensión de embebido $\times$ el número de *tokens* en el vocabulario es igual al número de **pesos** o **parámetros** del modelo.
![[embedding-example.png]]

# Algoritmos
- [[word2vec]]
- [[fasttext]]