- #cnn | #undersampling | #outliers | #ml

# Concepto
- Se intenta eliminar zonas de alta densidad de la clase mayoritaria, de forma que las instancias restantes están más juntas entre sí y por tanto el algoritmo de predicción puede ser capaz de encontrar patrones en esas zonas del espacio.

# Algoritmo
- Se escogen todas las instancias de la clase minoritaria y solo una de la clase mayoritaria.
- A continuación se clasifican todas las instancias del *dataset* original utilizando el nuevo *dataset* como un conjunto de entrenamiento y el *dataset* como conjunto de *test*.
- Todas las instancias bien clasificadas se eliminan del conjunto original y todas aquellas mal clasificadas se quedan (con su clase original correspondiente).
![[cnn-undesampling-example.png]]
- Es una especie de [[k-nearest neighbors|knn]], de forma que inicialmente se pueden escoger $c$ instancias de la clase mayoritaria para mantener. El ejemplo explicado es con $1$-NN.