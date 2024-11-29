"Dada la siguiente matriz:
laberinto <- matrix(c(
\"O\", \"X\", \"O\", \"O\", \"O\",
\"O\", \"X\", \"X\", \"X\", \"O\",
\"O\", \"O\", \"O\", \"O\", \"X\",
\"X\", \"X\", \"X\", \"O\", \"X\",
\"O\", \"O\", \"O\", \"O\", \"O\"
), nrow = 5, byrow = TRUE)

Indique donde están las \"X\" usando la función which
Indique donde están las \"X\" usando la función which pero usando el parámetro arr.ind
Indique cuantos valores \"X\" hay en la matriz
Reemplace las \"X\" por 1 y los \"O\" por 0 y convierta la matrix en una matrix numérica
Indique si la matrix es simétrica
Cree una nueva matrix con los valores en las filas y columnas pares"

laberinto <- matrix(c(
  
  "O", "X", "O", "O", "O",
  
  "O", "X", "X", "X", "O",
  
  "O", "O", "O", "O", "X",
  
  "X", "X", "X", "O", "X",
  
  "O", "O", "O", "O", "O"
  
), nrow = 5, byrow = TRUE)

which(laberinto == "X")
which(laberinto == "X", arr.ind = TRUE)
sum(laberinto == "X")
laberinto[laberinto == "X"] <- "1"
laberinto[laberinto == "O"] <- "0" # Otra opción sería con ifelse
laberinto <- matrix(as.numeric(laberinto), nrow = nrow(laberinto))
all(laberinto == t(laberinto))
new_matrix <- laberinto[seq(2, nrow(laberinto), 2), seq(2, ncol(laberinto), 2)]
new_matrix

"Crea una matriz que represente un Sudoku. Debe estar inicializada en NA. 
Luego rellenar solo un 10% de las casillas con un valor entre 1 y 9 
(HINT: usa la función sample). No se preocupe de que sea un esquema válido 
(se pueden repetir valores en una fila por ejemplo... eso ya lo 
haremos más adelante...)"

sudoku <- matrix(NA, nrow = 9, ncol = 9) # Matrix 9x9 de NaN
num_to_fill <- round(0.1 * (nrow(sudoku) * ncol(sudoku))) # 10% de posiciones

# Se rellena un 10% del sudoku con números entre 1 al 9
values_to_fill <- sample(1:9, num_to_fill)

# Hay 9x9 posiciones posibles, las escogemos aleatoriamente
positions <- sample(1:(nrow(sudoku)*ncol(sudoku)), num_to_fill)

# Rellenamos las posiciones del sudoku (de manera lineal, no por fila columna)
sudoku[positions] <- values_to_fill  
  
"Teniendo estas dos matrices:
matriz <- matrix(sample(1:9, 30,replace=TRUE), nrow=5)
mascara <- matrix(sample(c(TRUE,FALSE), 30, replace=TRUE), nrow=5)
  1- Aplique la máscara a la matrix, en donde si la posicion en la máscara es 
  FALSE entonces el resultado es 0 y sino se mantiene su valor
  2- Calcule el producto de la matriz original por la matriz modificada 
  traspuesta (NOTA: la matriz final será de 5x5)"
  
matriz <- matrix(sample(1:9, 30,replace=TRUE), nrow=5)
mascara <- matrix(sample(c(TRUE,FALSE), 30, replace=TRUE), nrow=5)

matriz_mod = matriz
matriz_mod2 = matriz
matriz_mod[mascara == FALSE] <- 0 # Primera forma
matriz_mod2 = matriz * mascara # Segunda forma
result <- matriz %*% t(matriz_mod)

"Crea un array 5D llamado \"mi_array5D\" con las siguientes características, rellénalo con números enteros consecutivos comenzando desde 1:
    Tamaño de la primera dimensión: 2 elementos
    Tamaño de la segunda dimensión: 3 elementos
    Tamaño de la tercera dimensión: 4 elementos
    Tamaño de la cuarta dimensión: 5 elementos
    Tamaño de la quinta dimensión: 6 elementos
 Accede a los siguientes elementos del array:
    los elementos pares (analice la posición de esos valores en las 5 dimensiones)
    estudiando el caso anterior muestre los elementos impares sin usar ningun tipo de condicional, igualdad o desigualdad
 Redimensiona el array para que tenga:
    Tamaño de la primera dimensión: 6 elementos.
    Tamaño de la segunda dimensión: 5 elementos.
    Tamaño de la tercera dimensión: 4 elementos.
    Tamaño de la cuarta dimensión: 3 elementos.
    Tamaño de la quinta dimensión: 2 elementos."

dim <- c(2,3,4,5,6)
num_elementos <- prod(dim)
mi_array5D <- array(1:num_elementos, dim)
pares <- mi_array5D[mi_array5D %% 2 == 0]
which(mi_array5D %% 2 == 0, arr.ind = TRUE)

# En la primera dimensión todos los pares están en el índice 2, en la segunda
# dimensión las posiciones de los pares son cíclicas de 1 a 3. Los ciclos 
# continuan en las siguientes dimensiones, pero los saltos se hacen más grandes.
# dim 3 -> 111222333, dim 4 -> 1111111..222222..333333 etc.

impares <- pares + 1 # Todos los pares + 1 muestran todos los impares (excepto el 1)
impares

mi_nuevo_array5D <- array(mi_array5D, dim = c(6,5,4,3,2))

"Supongamos que tenemos un grafo no dirigido con 6 nodos etiquetados del 1 al 6,
y las siguientes aristas:
    Arista 1: (1, 2)
    Arista 2: (1, 3)
    Arista 3: (2, 4)
    Arista 4: (2, 5)
    Arista 5: (3, 6)
    Arista 6: (4, 5)
Crea una matriz de adyacencia \"matriz_adyacencia\" en R para representar este 
grafo. En una matriz de adyacencia, las filas y columnas representan los nodos, 
y un valor 1 indica que existe una arista entre los nodos correspondientes, 
mientras que un valor 0 indica la ausencia de una arista.
    Calcula el grado del nodo 4 en el grafo. El grado de un nodo es la cantidad 
    de aristas que inciden en él
    Encuentra todos los nodos adyacentes del nodo 2
    Cuenta la cantidad de nodos que tienen una arista hacia si mismos (self-loop)"

grafo <- matrix(0, nrow = 6, ncol = 6)
edges <- matrix(c(
  1, 2,
  1, 3,
  2, 4,
  2, 5,
  3, 6,
  4, 5
), ncol = 2, byrow = TRUE)
grafo[edges] <- 1
grafo
sum(grafo[4,]) # Grado del nodo 4 (sería igual si se calcula por columnas)
# Los nodos adyacentes del nodo dos son:
which(grafo[2,] == 1, arr.ind = TRUE) # De nuevo, funciona si lo haces por cols
sum(diag(grafo)) # Ninguno
