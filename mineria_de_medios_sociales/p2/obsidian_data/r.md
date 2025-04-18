- #r | #statistics | #plots | #datascience

# Tips
- Los elementos dentro de las estructuras de datos en $R$ empiezan por el índice $1$ y no por $0$.
- El operador de resta excluye elementos de una estructura de datos. Ej: `1:6[-2]` es `1 3 4 5 6`. Si hacemos `[-x:-y]` se excluirán todo el rango de valores desde el índice `x` hasta el índice `y`.
- Para obviar los valores nulos en una estructura al hacer una operación, usar `na.rm = TRUE`. Se puede usar en `sum, prod, mean, etc`.
- Los conjuntos `letters` y `LETTERS` contienen las letras del alfabeto en minúsculas y mayúsculas correspondientemente.
- Los conjuntos `month.name` y `month.abb` devuelven los meses, el primero con los nombres originales en inglés y el segundo con sus abreviaturas.

# Operadores aritméticos y lógicos
- Suma -> `1+5`
- Resta -> `5-4`
- Multiplicación -> `5*2.5`
- División -> `3/ (2+1)`
- Potencia -> `2^5` o `2**5`
- Módulo -> `10%%4`
- Demás operadores de comparación como $\ge, \le, >, <, ==, !=$
- Operadores lógicos -> AND es `&`, OR es `|`
- Pertenencia -> `%in%`

# Funciones
- `all` -> Comprueba que todos los elementos de un vector cumplen la condición lógica. Ej: `all(c(TRUE, TRUE))` es `TRUE`, `all(c(1,2,-4) > 0)` es `FALSE`.
- `intersect` -> Devuelve un vector con la intersección de los dos vectores pasados por parámetro.
- `union` -> Devuelve un vector con la unión de los dos vectores pasados por parámetros.
- `setdiff` -> Devuelve un vector con los elementos de $X$ que no están en $Y$ de los dos vectores pasados por parámetro. 
- `length` -> Longitud de una estructura de datos.
- `nchar` -> Longitud de una cadena de caracteres.
- `prod, min, max, sum` -> Devuelven el producto, el valor mínimo, el valor máximo y la sumatoria de los vectores a los que se aplique.
- `seq` -> Crea vectores desde `from` hasta `to` usando pasos de tamaño especificado en `by` y que por defecto son de uno en uno.
- `rep` -> Repite la secuencia que se le dé el número de veces que se especifique.
- `is.na, is.numeric, etc` -> Funciones de este estilo devuelven si lo que se le pasa es del tipo de dato que se comprueba. 
- `as.numeric, as.character, as.logical, etc` -> Convierte lo que se la pase el tipo de dato especificado.
- `which` -> Devuelve los índices de la estructura donde se han encontrado aquellos valores que cumplen una condición lógica. Si se le especifica `arr.ind = TRUE` devuleve índices de fila y columna en caso de ser una estructura de tipo matriz.
- `names` -> Se usa para asignar o acceder a los nombres de los elementos de un objeto, como un vector, lista, o *data frame.*
- `unlist` -> Devuele un vector de una lista que se va a descomponer. Por defecto se conservan los nombres de los elementos originales y por defecto se hace la descomposición de manera recursiva (por si existen listas que contengan listas).
- `colnames` -> Se utiliza para obtener o asignar los nombres de las columna de un objeto, como un *data frame* o una matriz.
- `dim, nrow, ncol, diag` -> Devuelve el número de dimensiones de un objeto, el número de filas, el número de columnas y la diagonal de un objeto.
- `table` -> Se utiliza para crear tablas de contingencia o frecuencias, mostrando el conteo de ocurrencias de los niveles de una variable categórica.
- `sort` -> Ordena los valores de un vector. Si `decreasing` está a `TRUE`entonces su orden será descendente. Si `na.last` es `TRUE`, los valores `NA` se colocarán al final, si es `FALSE` al principio y si es `NA` se excluyen.
- `summary` -> Resumen estadístico del objeto pasado por parámetro. Suele ser más útil con *data frames*.
- `rbind` -> Se utiliza para combinar *data frames* o matrices por filas, es decir, agrea nuevas filas a un objeto existente. 
- `runif` -> Genera un vector con variables aleatorias entre $[x,y]$. Se especifica primero longitud del vector, luego mínimo y máximo.
- `merge` -> Se utiliza para combinar dos data frames o tablas basándose en una o más columnas clave (también conocidas como columnas de unión). Es especialmente útil para realizar uniones de tipo SQL, permitiendo combinar datos de diferentes fuentes de manera estructurada. En el parámetro `by` se especifica la columna por la que se hace la unión. Si es una unión externa completa, se pone `all=TRUE`, si es una unión externa izquierda se pone `all.x=TRUE` y si es derecha `all.y=TRUE` (mirar [[sql join]]).
- `cut` -> Crea un factor definiendo los intervalos y las etiquetas de cada intervalo. Se puede poner si el resultado deber ser un factor ordinal (ordenado).
 
# Estructuras de datos
## Vectores
- Secuencia de elementos del mismo tipo.
- Creación: `c()` o `vector()`.
- Tipos: numérico, carácter, lógico.
- Tipo de dato **homogéneo**.
- Ejemplo: `numeros <- c(1, 2, 3, 4, 5)`.
- Si se crea un vector que contenga elementos de varios tipos de datos, el vector transformará los datos al tipo más inclusivo posible:
	- Se transformará a un vector de *integers* si hay valores lógicos y enteros, a *numeric* si incluye decimales, enteros o lógicos y a *character* si se incluye además de eso una cadena de caracteres.
- Se pueden crear vectores usando un rango de valores -> `my_vector <- 1:4.
- Si se utilizan operadores entre vectores, las operaciones se realizarán elemento a elemento. Por ejemplo: `1:3 < 2:4` es `TRUE TRUE TRUE`.
- Se pueden filtrar los elementos de un vector utilizando el operador `[]` y una condición lógica. Ej: `c[c < 2 & c > -4]` devuelve los elementos de `c` que sean menores a $2$ y mayores a $-4$. Esto se puede usar para el filtrado de elementos con una máscara, donde la máscara es un vector de *booleanos* y de tamaño igual al vector filtrado.
## Matrices
- Arreglo bidimensional de elementos del mismo tipo.
- Creación: `matrix()`.
- Tipo de dato **homogéneo**.
- Ejemplo: `mi_matriz <- matrix(1:9, nrow = 3, ncol = 3)`.
- Si se especifica `byrow` los elementos de la matriz se crearán de izquierda a derecha y de arriba a abajo. En caso de ser `FALSE` entonces se crearán de arriba a abajo y de izquierda a derecha.
- Cuando seleccionas una sola fila o columna de una matriz o data frame, el uso de `drop = TRUE` eliminará la dimensión adicional, devolviendo un vector en lugar de una matriz de una sola fila o columna.
- Los operados tradicionales funcionan *element-wise* con las matrices. Para multiplicar matricialmente -> `%*%`.
## Arrays
- Generalización de matrices a más dimensiones.
- Creación: `array()`.
- Tipo de dato **homogéneo**.
- Ejemplo: `mi_array <- array(1:24, dim = c(2, 3, 4))`.
## Listas
- Colección de elementos de diferentes tipos.
- Creación: `list()`.
- Tipo de datos **heterogéneo**.
- Ejemplo: `mi_lista <- list(nombre = "Juan", edad = 30, scores = c(95, 87, 92))`.
- Para acceder al elemento en el índice $X$ se debe usar: `mi_lista[[X]]`. Por defecto, al acceder con un solo par de corchetes se devuelve un subconjunto de ls lista conteniendo el elemento en la posición $X$, pero sigue siendo una lista.
## Data Frames
- Tabla de datos con columnas de diferentes tipos.
- Similar a una hoja de cálculo.
- Cada columa es **homogénea** en cuanto a los tipos de datos, entre distintas columnas el tipo de dato es **heterogéneo**.
- Creación: `data.frame()` o `read.csv()` u otro.
- Ejemplo:
```r
  df <- data.frame(
    nombre = c("Ana", "Carlos", "Eva"),
    edad = c(25, 30, 28),
    ciudad = c("Madrid", "Barcelona", "Sevilla")
  )
```
## Factores
- Los factores son vectores para variables categóricas.
- Estos constan de las *labels*, que son los valores categóricos, y de los *levels*, que son los distintos valores que pueden tomar.
- Para realizar cálculos con ellos se pueden convertir a números, lo que convertirá los niveles a sus índices.
- Ejemplo: `my_factor <- factor(c("Rojo", "Azul", "Verde"), levels = c(1,2,3))`.
- Para variables categóricas ordinales es conveniente poner el parámetro `orderes = TRUE`.
- Si te indexan algunos valores de un factor, pero no se requieren todos los niveles, solo los de aquellos elementos seleccionados, ha de especificarse el parámetro`drop=TRUE`. Ejemplo: `factor(c("AA", "BA", "CA"))[1:2, drop=TRUE]`. De otra forma devolverá los valores seleccionados, pero todos los niveles.

# Gráficas
- [[ggplot]]

# Tidyverse
- [[tidyverse]]

# Strings
- [[strings in r]]

# Loops (apply)
- Las funciones de la familia *apply* son bucles implícitos.

| Función      | Entrada             | Salida                 | Comentario                                        | Ejemplo                                                                                                                      |
| ------------ | ------------------- | ---------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `apply`      | Matriz o array      | Vector, array o lista  | Aplica una función a filas o columnas             | `apply(matrix(1:9, nrow=3), 1, sum)` - Suma de cada fila en la matriz                                                        |
| `lapply`     | Lista o vector      | Lista                  | Aplica una función y devuelve una lista           | `lapply(1:5, function(x) x^2)` - Cuadrado de cada número en la lista                                                         |
| `sapply`     | Lista o vector      | Vector, matriz o lista | Simplificación automática                         | `sapply(1:5, function(x) x^2)` - Devuelve un vector con los cuadrados                                                        |
| `vapply`     | Lista o vector      | Vector, matriz o lista | Simplificación segura                             | `vapply(1:5, function(x) x^2, numeric(1))` - Devuelve un vector numérico con los cuadrados                                   |
| `tapply`     | Datos y categorías  | Array o lista          | Aplicación sobre categorías ("ragged")            | `tapply(c(1, 2, 3, 4), c("A", "A", "B", "B"), sum)` - Suma de elementos por categoría                                        |
| `mapply`     | Listas y/o vectores | Vector, matriz o lista | Aplicación múltiple                               | `mapply(sum, 1:5, 6:10)` - Suma elemento a elemento de dos vectores                                                          |
| `rapply`     | Lista               | Vector o lista         | Aplicación recursiva                              | `rapply(list(a=1, b=list(c=2, d=3)), function(x) x*2)` - Multiplica por 2 cada elemento, incluyendo listas anidadas          |
| `eapply`     | Entorno             | Lista                  | Aplica una función a objetos en un entorno        | `eapply(.GlobalEnv, is.numeric)` - Comprueba si cada objeto en el entorno global es numérico                                 |
| `dendrapply` | Dendograma          | Dendograma             | Para dendogramas                                  | `hc <- hclust(dist(USArrests)); dendrapply(as.dendrogram(hc), function(x) x)` - Aplica una función a nodos de un dendrograma |
| `rollapply`  | Datos               | Similar a la entrada   | Del paquete `zoo` para datos de series temporales | `library(zoo); rollapply(zoo(1:10), 3, sum)` - Suma móvil de 3 elementos en una serie temporal                               |
