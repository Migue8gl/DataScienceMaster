"Crea un factor llamado fact_puntuacion que contenga puntuaciones del 1 al 5
para 1000 productos. Utiliza summary para obtenet un resumen de la distribución
de puntuaciones ¿Qué puedes deducir del tipo de aleatoriedad usada en
sample? ¿Es normal, uniforme?"

fact_puntuacion <- factor(
    x = sample(x = 1:5, size=10000, replace = TRUE), 
    levels=1:5
  )
summary(fact_puntuacion)
# Tal y como muestra en terminal, parece que sample utiliza una distribución
# uniforme. Cada nivel tiene más o menos el mismo número de elementos.

"Crea un factor ordenado para edades entre 0 y 100 de 100 personas. Utiliza
la función cut para generar 3 niveles separados. Ahora reetiqueta el factor
como joven, adulto o anciano. Muestra a la frecuencia de cada nivel."

edades <- sample(0:100, 100, replace = TRUE)
edades_factor <- cut(
    edades, 
    breaks = c(0, 18, 70, 100), 
    labels = c("joven", "adulto", "anciano"),
    ordered_result = TRUE,
    include.lowest = TRUE
  )
summary(edades_factor)

"Crea un factor para el siguiente vector teniendo en cuenta que los niveles
posibles son todas las letras de la A a la Z.
  - Modifica una de las I por una F
  - Agrega una nueva A en última posición
  - Indica las letras que aparecen en el factor ordenadas por su frecuencia, 
    empezando por la menos frecuente."

v <- c("A", "I", "I", "J", "A", "Z", "X")
factor_letters <- factor(v, levels = LETTERS[1:26])
factor_letters[2] <- "F"
factor_letters[length(factor_letters) + 1] = "A"
sort(summary(factor_letters), decreasing = TRUE)





