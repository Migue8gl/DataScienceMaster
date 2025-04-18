- #specialtypesmatrixes

### Matriz cuadrada
- Una matriz cuadrada es aquella que tiene igual número de filas y de columnas.

### Matriz identidad
- La matriz identidad es aquella compuesta por todo $0's$ excepto en la diagonal, la cual se compone de $1's$. La matriz identidad es el equivalente del **elemento neutro** dentro del producto de matrices.
- Se compone por los [[linear transformations#Vectores base|vectores base]].

### Matriz inversa
- La matriz inversa de una matriz $A$ se escribe como $A^{-1}$ y su función es la de deshacer la transformación aplicada por la matriz $A$.
- Si multiplicamos la matriz $A$ por su matriz inversa, obtendremos la matriz identidad.

### Matriz diagonal
- Las matrices diagonales son aquellas que se componen por $0's$ a excepción de su diagonal.
- Son interesantes en ciertas operaciones computacionales porque en esencia son simples escalares aplicados a un espacio vectorial.

### Matriz triangular
- Parecidas a las matrices diagonales. De la diagonal hacia uno de los dos lados (arriba o abajo) nos encontramos con valores distintos de cero, el resto son ceros.
- Son deseables en ciertas tareas de análisis numérico, pues son más fáciles de resolver en sistemas de ecuaciones.

### Matrices dispersas
- Son matrices cuya mayoría de elementos son iguales a cero, con solo unos pocos valores distintos a cero.
- Desde un punto de vista computacional son eficientes en memoria. Si tenemos una matriz conformada por muchos ceros, no es necesario guardarlos todos en memoria, solo es preciso guardar aquellas celdas e índices cuyo valor sea distinto de cero.
