"Crear una función \"creciente\" que indique si los elementos de un array dado
son estrictamente crecientes. No se permite ordenar el vector."

creciente <- function(arr) {
  all(head(arr, -1) < tail(arr, -1))
}

creciente(c(1, 2, 3, 4))
creciente(c(1, 3, 2, 4))
creciente(c(5, 6, 7, 8, 9))
creciente(c(10, 9, 8, 7))

"Crear una función \"montecarlo\" que calcule la estimación de la integral
dada: ∫01x2dx"
montecarlo <- function(N) {
  r1 <- runif(N)
  r2 <- runif(N)
  hits <- sum(r2 < r1 ^ 2)
  return(hits / N)
}

montecarlo(10000)

"Crea una lista de cinco matrices numéricas y ordena cada una de ellas tras su
creación (el elemento [1,1] tendrá el valor menor y el [#filas,#columnas]
el valor mayor)."

create_and_sort_matrices <- function(rows, cols) {
  matrices_list <- lapply(1:5, function(x) {
    matriz <- matrix(runif(rows * cols), nrow = rows, ncol = cols)
    sorted_matriz <- matrix(sort(matriz), nrow = rows, ncol = cols)
    sorted_matriz
  })
  matrices_list
}

create_and_sort_matrices(3, 3)

"Calcula el valor mínimo de cada columna de una matriz, pero suponiendo que 
los números impares son negativos y los pares positivos."

m <- matrix(sample(1:15, 9), nrow = 3, ncol = 3)

odd_to_negative <- function(m) {
  indexes <- which(m %% 2 == 1, arr.ind = TRUE)
  m[indexes] <- m[indexes] * -1
  m
}

m <- odd_to_negative(m)
apply(m, 2, min)

"Dada una matriz devuelve una lista de todos los valores mayores que 7 
de cada fila."

m <- matrix(sample(1:15, 9), nrow = 3, ncol = 3)
lapply(1:nrow(m), function(i) m[i, m[i, ] > 7])


