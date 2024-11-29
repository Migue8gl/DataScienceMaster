"Crea una función \"impares\" que dada una matriz devuelva el número de 
elementos impares que contiene."

matriz <- matrix(sample(1:20, 16, replace = TRUE), nrow=4)

impares <- function(x) {
  length(x[which(x %% 2 != 0, arr.ind = TRUE)])
}

impares(matriz)

"Crear una función \"cambio\" que, dada una matriz de números enteros, devuelva 
una nueva matriz con todos los NA sustituidos por 0."

matriz <- matrix(sample(1:20, 16, replace = TRUE), nrow=4)
matriz[sample(1:20, 4, replace = TRUE)] <- NA

cambio <- function(x) {
  x[which(is.na(x), arr.ind = TRUE)] <- 0
  x
}
matriz <- cambio(matriz)

"Crear una función \"reducir\" que dados dos vectores devuelva una lista con 
dos componentes: 1) un vector con los elementos comunes sin repetir y 2) la 
cantidad de elementos totales eliminados al quitar los repetidos."

vec1 <- sample(1:99, 10, replace = TRUE)
vec2 <- sample(1:99, 10, replace = TRUE)

reducir <- function(x, y) {
  total_length <- length(x) + length(y)
  new_vec <- unique(x[x %in% y])
  list(new_vec, total_length - length(new_vec))
}

new_vec <- reducir(vec1, vec2)

"Crear una función \"vyc\" que dada una cadena de caracteres devuelva una lista 
de dos componentes, uno que contenga las vocales y otro las consonantes 
(en orden alfabético y minúsculas sin repetir)."

cadena <- "Eres una Persona muy InteresSanteeee"

vyc <- function(x) {
  x <- unlist(strsplit(tolower(x), ""))
  vocales <- c("a", "e", "i", "o", "u")
  x_vocales <- sort(unique(x[x %in% vocales]))
  x_cons <- sort(unique(x[!x %in% vocales & x %in% letters]))
  
  list(x_vocales, x_cons)
}
vyc(cadena)

"Crear una función \"subpos\" que dado un vector v y dos valores x e y (siendo 
y un parámetro opcional), devuelva una nuevo vector con los valores incluidos 
después de la aparición de la primera x (si x no está, empieza desde el 
principio) hasta la primera y (que aparezca después de la primera x), (si y no 
está o no se pasa por parámetro, termina hasta el final del vector)."

subpos <- function(v, x, y = NULL) {
  start_pos <- c(1, which(v == x, arr.ind = TRUE)[1])
  start <- is.na(start_pos[1]) + (!is.na(start_pos[1])) * start_pos[1]
  start_pos <- start_pos[!is.na(start_pos)]
  start <- start_pos[length(start_pos)]
  
  end_pos <- c(length(v), which(v == y, arr.ind = TRUE)[1])
  end_pos <- end_pos[!is.na(end_pos)]
  end <- end_pos[length(end_pos)]
  
  v[start:end]
}

subpos(1:4, 8, 3) 
subpos(1:4, 2, 8)
subpos(1:4, 2)

