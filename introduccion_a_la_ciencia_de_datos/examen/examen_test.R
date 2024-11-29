library(tidyverse)

df <- data.frame(
  nombre_estudiante = c(
    "Ana María",
    "Luis Alberto",
    "Carlos Enrique",
    "María Fernanda",
    "Elena Sofía",
    "Miguel Ángel",
    "Lucía",
    "Pablo",
    "Laura Isabel",
    "Juan Carlos",
    "José Antonio",
    "Rosa María",
    "Fernando",
    "Silvia",
    "Gabriel",
    "Patricia",
    "Francisco Javier",
    "Cristina",
    "Daniel",
    "Andrea"
  ),
  apellido_estudiante = c(
    "García López",
    "Martínez Torres",
    "Pérez Gómez",
    "González Fernández",
    "López Morales",
    "Ramírez García",
    "Moreno Sánchez",
    "Jiménez Muñoz",
    "Rodríguez Díaz",
    "Díaz Pérez",
    "Gutiérrez López",
    "Navarro García",
    "Hernández Castro",
    "Vargas López",
    "Lara Serrano",
    "Ruiz Gálvez",
    "Vega Muñoz",
    "Ortega Jiménez",
    "Castillo Romero",
    "Sánchez Martínez"
  ),
  asignatura = c(
    "Matemáticas",
    "Historia",
    "Inglés",
    "Latín",
    "Química",
    "Historia",
    "Matemáticas",
    "Inglés",
    "Química",
    "Historia",
    "Física",
    "Biología",
    "Geografía",
    "Economía",
    "Educación Física",
    "Arte",
    "Filosofía",
    "Informática",
    "Informática",
    "Matemáticas"
  ),
  género = factor(
    c(
      "F",
      "M",
      "M",
      "F",
      "F",
      "M",
      "F",
      "M",
      "F",
      "M",
      "M",
      "F",
      "M",
      "F",
      "M",
      "F",
      "M",
      "F",
      "M",
      "F"
    ),
    levels = c("F", "M")
  ),
  nota = c(
    4.5,
    6.0,
    7.5,
    3.0,
    8.5,
    5.5,
    4.0,
    6.5,
    9.0,
    5.0,
    7.8,
    6.3,
    5.7,
    8.0,
    4.2,
    7.1,
    6.9,
    8.8,
    10,
    7.6
  )
)

apellido <- df %>% filter(género == "F" &
                            nota < 5) %>% pull(apellido_estudiante)
str_remove(apellido, " .*")

top_k_asig <- df %>%
  group_by(asignatura) %>%
  summarise(total_estudiantes = n()) %>%
  arrange(desc(total_estudiantes)) %>%
  pull(asignatura) %>%
  .[1:5]

df %>%
  filter(asignatura %in% top_k_asig) %>%
  group_by(género, asignatura) %>%
  summarise(media = mean(nota), aprobados = sum(nota >= 5))

matriz <- matrix(sample(-10:10, 9), nrow = 3)
matriz[1, 3] <- NA
matriz

ejercicio <- function(mat) {
  a <- diag(mat[, ncol(mat):1])
  a <- a[!is.na(a) & a > 0]
  b <- apply(mat, 2, min, na.rm = TRUE)
  c(a=a,b=b)
}

ejercicio(matriz)


df <- df %>%
  mutate(aprobado = case_when(
    nota == 10 ~ "Matrícula",
    nota == 9 ~ "Sobresaliente",
    nota >= 7 & nota < 9 ~ "Notable",
    nota >= 5 & nota < 7 ~ "Aprobado",
    nota < 5 | is.na(nota) ~ "Suspenso",
  )) %>%
  mutate(aprobado = factor(aprobado))

ggplot(df, aes(x = aprobado)) +
  geom_bar() +
  facet_wrap(~ asignatura) +
  labs(x = "Calificación", y = "Número de alumnos") 

v <- c("hola","mundo")
w <- c(1,6)

res <- rbind(v, w)
res
flatten(as.list(res))
