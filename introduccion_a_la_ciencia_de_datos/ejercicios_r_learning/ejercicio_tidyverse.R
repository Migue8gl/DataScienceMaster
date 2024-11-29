"Utiliza el dataset mtcars y el paquete tidyverse para:
  - Mostrar las 5 primeras filas del dataset
  - Convertir la variables cyl, gear, carb en factores y cambiar las variables
    vs y am en lógicas y mostrar la estructura del dataset.
  - Mostrar solo los coches con una potencia (hp) mayor a 100
  - Seleccionar solo las columnas mpg, cyl, hp y qsec del dataset
  - Calcular la cantidad total de coches para cada valor único en la columna cyl
  - Encontrar el modelo de coche con la mayor potencia (hp) y mostrar su
    información completa
  - Calcular el promedio de potencia de los coches con 8 cilindros"

library(tidyverse)
mtcars %>% head(5)
mtcars <- mtcars %>%
  mutate(
    cyl = as.factor(cyl),
    gear = as.factor(gear),
    carb = as.factor(carb),
    vs = as.logical(vs),
    am = as.logical(am),
  )
mtcars %>% head(5)
mtcars %>% filter(hp > 100)
mtcars %>% select(mpg, cyl, hp, qsec)
mtcars %>% count(cyl)
mtcars %>% filter(hp == max(hp))
mtcars %>% filter(cyl == 8) %>% summarise(avg = mean(hp))

"Supongamos que tienes un data frame llamado lego_sets que contiene información
sobre diferentes conjuntos de Lego, incluyendo el nombre del conjunto,
el número de piezas, el tema y el año de lanzamiento:
    - Filtra y muestra solo los conjuntos de Lego lanzados en el año 2020
    - Encuentra y muestra el nombre y el número de piezas del conjunto de Lego
      más grande
    - Calcula la cantidad total de piezas para cada tema y muestra un resumen
      ordenado en orden descendente
    - Calcula cuántos conjuntos de Lego se lanzaron en cada año y muéstralo
      ordenado por año
    - Encuentra los 3 temas más populares (con más conjuntos) y muestra el
      número de conjuntos y el número total de piezas para cada uno de ellos"

nombres <- paste("Conjunto", 1:80)
piezas <- sample(50:1000, 80, replace = TRUE)
temas <- sample(
  c(
    "Ciudad",
    "Espacio",
    "Arquitectura",
    "Granja",
    "Dinosaurios",
    "Aventuras",
    "Piratas"
  ),
  80,
  replace = TRUE
)
años <- sample(2000:2023, 80, replace = TRUE)

lego_sets <- data.frame(
  Set_Name = nombres,
  Piece_Count = piezas,
  Theme = temas,
  Year = años
)

lego_sets %>% filter(Year == 2020)
lego_sets %>% filter(Piece_Count == max(Piece_Count)) %>%
  summarise(Name = Set_Name, Count = Piece_Count)
lego_sets %>%
  group_by(Theme) %>%
  summarize(Count = sum(Piece_Count)) %>%
  arrange(desc = Count)
lego_sets %>%
  group_by(Year) %>%
  summarize(Count = n()) %>%
  arrange(desc = Year)
lego_sets %>%
  group_by(Theme) %>%
  summarize(Count = n(), Pieces = sum(Piece_Count)) %>%
  top_n(3)

"Dado un data frame llamado bebidas que contiene informsación sobre la edad,
género, tipo de bebida y cantidad de copas consumidas por un grupo de personas,
filtra y muestra únicamente a las mujeres mayores de 20 años.
Para este suconjunto calcula la media, el máximo, la cantidad de copas
totales y la desviación estándar de la edad para cada combinación de tipo de
bebida. Finalemnte, agrega una nueva columna que indique cuántas personas se
encuentran en cada grupo de tipo de bebida."

num_elementos <- 100
edades <- sample(18:70, num_elementos, replace = TRUE)
sexos <- sample(c("Hombre", "Mujer"), num_elementos, replace = TRUE)
tipos_bebida <- sample(c("Cerveza", "Vino", "Refresco", "Cóctel", "Agua"),
                       num_elementos,
                       replace = TRUE)
cantidad_copas <- sample(1:5, num_elementos, replace = TRUE)

bebidas <- data.frame(
  Edad = edades,
  Sexo = sexos,
  Bebida = tipos_bebida,
  Copas = cantidad_copas
)

mujeres_mayores_20 <- bebidas %>%
  filter(Sexo == "Mujer", Edad > 20)

estadisticas_bebidas <- mujeres_mayores_20 %>%
  group_by(Bebida) %>%
  summarise(
    Media_Edad = mean(Edad),
    Maximo_Edad = max(Edad),
    Total_Copas = sum(Copas),
    Desviacion_Estandar_Edad = sd(Edad),
    Cantidad_Personas = n()
  )
estadisticas_bebidas

"Dado un data frame llamado peliculas, indique los 2 géneros con mayor
puntuación media por país pero solo de peliculas para mayores de 13 o
superiores, que incluya el número de películas en ese género, su beneficio
medio (ganancia-presupuesto), la desviación estándar de la puntuación y la
puntuación media. Ordenar los resultados por puntuación media por país de
forma descendente."

num_filas <- 150

set.seed(123)  # Para reproducibilidad
paises <- sample(c("EE. UU.", "Reino Unido", "Francia", "España", "Italia"),
                 num_filas,
                 replace = TRUE)
nombres <- paste("Película", 1:num_filas)
años <- sample(1980:2023, num_filas, replace = TRUE)
puntuaciones <- round(runif(num_filas, 1, 10), 1)
tematicas <- sample(c("Acción", "Drama", "Comedia", "Ciencia Ficción", "Animación"),
                    num_filas,
                    replace = TRUE)
directores <- paste("Director", 1:num_filas)
companias <- sample(
  c(
    "Warner Bros.",
    "Universal Pictures",
    "Disney
",
    "Sony Pictures",
    "Paramount Pictures"
  ),
  num_filas,
  replace = TRUE
)
presupuestos <- round(runif(num_filas, 1000000, 50000000), 2)
ganancias <- round(presupuestos * runif(num_filas, 0.5, 2.5), 2)
ratings <- sample(c("G", "PG", "PG-13", "R", "NC-17"), num_filas, replace = TRUE)  # Columna de rating ficticio

peliculas <- data.frame(
  País = paises,
  Nombre = nombres,
  Año = años,
  Puntuación = puntuaciones,
  Temática = tematicas,
  Director = directores,
  Compañía = companias,
  Presupuesto = presupuestos,
  Ganancia = ganancias,
  Rating = ratings
)

peliculas %>%
  filter(Rating %in% c("R", "NC-17")) %>%
  group_by(País, Temática) %>%
  summarise(
    N_Pel_Genero = n(),
    Media_Ganancia = mean(Ganancia - Presupuesto, na.rm = TRUE),
    Puntuación_Desv_Estándar = sd(Puntuación, na.rm = TRUE),
    Puntuación_Media = mean(Puntuación, na.rm = TRUE)
  ) %>%
  top_n(2, Puntuación_Media) %>%
  arrange(País, desc(Puntuación_Media))

"Dado un conjunto de datos llamado notas que contiene información sobre las
notas de los estudiantes en varias asignaturas, agrega una columna con la nota
promedio de cada estudiante. Previamente reemplaza los valores faltantes con 0
(HINT: usa replace_na)."

estudiantes <- paste("Estudiante", 1:50)
notas_matematicas <- sample(c(NA, 5, 6, 7, 8, 9, 10), 50, replace = TRUE)
notas_historia <- sample(c(NA, 4, 5, 6, 7, 8), 50, replace = TRUE)
notas_ciencias <- sample(c(NA, 6, 7, 8, 9, 10), 50, replace = TRUE)

notas <- data.frame(
  Estudiante = estudiantes,
  Nota_Matemáticas = notas_matematicas,
  Nota_Historia = notas_historia,
  Nota_Ciencias = notas_ciencias
)

notas <- notas %>% replace_na(list(
  Nota_Ciencias = 0,
  Nota_Matemáticas = 0,
  Nota_Historia = 0
)) %>%
  mutate(Nota_Media = rowMeans(
    select(., Nota_Matemáticas, Nota_Ciencias, Nota_Historia), # Placeholder
    na.rm = TRUE
  ))
notas
