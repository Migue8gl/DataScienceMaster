"Usando el dataset mtcars:
  - Muestra las primeras 5 filas
  - Presenta un resumen del datset
  - Haz una copia del dataset y:
    - Convierte las variables cyl, gear y carb en factores
    - Convierte las variables vs y am en lógicas
    - Muestra solo los coches con una potencia hp mayor a 100
    - Crea un nuevo dataframe llamado coches_seleccionados que incluya solo
      las columnas mpg, cyl, hp, qsec
    - Calcula el resumen estadístico de la columna pmg para todos los coches
    - Idem para cyl
    - Calcula la cantidad total de coches para cada valor único en la columna
      cyl
    - Encuentra el modelo de coche con la mayor potencia y muestra su 
      información completa
    - Calcula el promedio de potencia de los coches con 8 cilindros"

summary(mtcars)
mtcars_copy <- mtcars
mtcars_copy$cyl <- factor(mtcars_copy$cyl)
mtcars_copy$gear <- factor(mtcars_copy$gear)
mtcars_copy$carb <- factor(mtcars_copy$carb)
mtcars_copy$vs <- mtcars_copy$vs == TRUE 
mtcars_copy$am <- mtcars_copy$am == TRUE
potencia_mayor_100 <- mtcars_copy[mtcars_copy$hp > 100,]
potencia_mayor_100
coches_seleccionados <- data.frame(
  mtcars_copy$mpg, 
  mtcars_copy$cyl, 
  mtcars_copy$hp,
  mtcars_copy$qsec
  )

summary(mtcars_copy$mpg)
mean(mtcars_copy$mpg)
median(mtcars_copy$mpg)
sd(mtcars_copy$mpg)
var(mtcars_copy$mpg)

# Obviamente va a fallar para datos categóricos
summary(mtcars_copy$cyl)
mean(mtcars_copy$cyl)
median(mtcars_copy$cyl)
sd(mtcars_copy$cyl)
var(mtcars_copy$cyl)

summary(mtcars_copy$cyl)

mtcars_copy[mtcars_copy$hp == max(mtcars_copy$hp),]
mean(mtcars_copy[mtcars_copy$cyl == 8,]$hp)

"Usando el dataset Cars93 del paquete MASS y el dataset mtcars:
  - Agrega una columna al datset de mtcars que contenga la marca y el modelo de
    cada coche
  - Combina los dos dataset por modelo y marca del coche solo para aquellos
    modelos presentes en ambos datasets. ¿Coinciden los demás valores?"

library(MASS)
mtcars_new <- mtcars
mtcars_new$Make <- NA
matched_indices <- match(rownames(mtcars_new), Cars93$Make)
mtcars_new$Make[!is.na(matched_indices)] <- levels(Cars93$Make[!is.na(matched_indices)])
merge(mtcars_new, Cars93, by="Make")

"Supongamos que tienes dos datafraj¡mes con información sobre estudiantes de
dos cursos diferentes. Cada dataframe contiene tres columnas: Nombre, Edad, 
Promedio.
  - Combina los dos dataframes en uno solo llamado todos_los_estudiantes
  - Agrega una nueva columna llamada Curso al dataframe todos_los_estudiantes
    que indica a qué curso pertenencen cada estudiante (A o B)"

curso_a <- data.frame(
  Nombre = c("Juan", "María", "Carlos"),
  Edad = c(20, 22, 21),
  Promedio = c(85, 92, 78)
)

curso_b <- data.frame(
  Nombre = c("Ana", "Luis", "Sofía", "Lourdes"),
  Edad = c(19, 23, 20, 21),
  Promedio = c(88, 90, 85, 80)
)

todos_los_estudiantes <- rbind(curso_a, curso_b)
matched_indices_a <- match(curso_a$Nombre, todos_los_estudiantes$Nombre)
matched_indices_b <- match(curso_b$Nombre, todos_los_estudiantes$Nombre)
todos_los_estudiantes$Curso <- NA
todos_los_estudiantes$Curso[matched_indices_a] = "A"
todos_los_estudiantes$Curso[matched_indices_b] = "B"



