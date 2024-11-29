library(tidyverse)
library(stringr)

"Dado un data frame df con columnas: nombre_estudiante (tipo character),
apellido_estudiante (tipo character), asignatura (tipo character), género
(factor, niveles [F,M]) y nota (tipo numeric) se pide que, usando funciones del
paquete tidyverse y stringr:
  A. Muestre el primer apellido (si este es compuesto o todo el apellido si
  solo tiene uno) de aquellas alumnas suspensas (nota menor a 5).
  B. Calcule la nota media y la cantidad de aprobados por género y asignatura,
  pero sólo de aquellas 5 asignaturas con mayor número de estudiantes."

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

df %>%
  filter(nota < 5, género == "F") %>%
  mutate(first_surname = str_remove(apellido_estudiante, " .*")) %>%
  pull(first_surname)

top_k_asignaturas <- df %>%
  group_by(asignatura) %>%
  mutate(total_alumnos_asignatura = n()) %>%
  ungroup() %>%
  arrange(desc(asignatura)) %>%
  pull(asignatura) %>%
  unique() %>%
  .[1:5]

df %>%
  filter(asignatura %in% top_k_asignaturas) %>%
  group_by(género, asignatura) %>%
  summarize(media = mean(nota), aprobados = sum(nota >= 5))

"Escriba una función que dada una matriz numérica cuadrada mat (teniendo valor
por omisión una matriz de 2x2 rellena con NA) retorne una lista con los
siguientes componentes (llamados A y B):
  A. Los elementos positivos de la diagonal secundaria (un NA no se considera
  positivo).
  B. Un vector con los valores mínimos de cada columna, excluyendo los NAs."

my_function <- function(mat = matrix(NA, nrow = 2, ncol = 2)) {
  a <- diag(mat)
  a <- a[!is.na(a)]
  
  b <- apply(mat, 2, min, na.rm = TRUE)
  
  list(a = a, b = b)
}
matriz <- matrix(sample(1:10, 9), nrow = 3)
matriz[1, 1] <- NA
matriz
my_function(matriz)


"Dado el data frame resultante del df del primer ejercicio agregue una nueva
columna aprobado de tipo factor ordenado que indique si el estudiante tiene
matricula de honor (si su nota es 10), sobresaliente (si su nota es mayor o
igual a 9), notable (entre 7 y 9), aprobado (entre 5 y 7) o suspenso (menor a 5
o NA). Utilizando este nueva columna realice un (único) gráfico de diagrama de
barras mediante el paquete ggplot2 de las notas, facetando el gráfico por cada
asignatura. Asegúrese que el gráfico tenga claramente indicados los ejes."

df <- df %>%
  mutate(aprobado = factor(ifelse(
    nota == 10,
    "Matricula",
    ifelse(
      nota >= 9,
      "Sobresaliente",
      ifelse(
        nota >= 7 & nota < 9,
        "Notable",
        ifelse(nota >= 5 &
                 nota < 7, "Aprobado", "Suspenso")
      )
    )
  )))

ggplot(df, aes(x = aprobado)) +
  geom_bar() +
  facet_wrap(~ asignatura) +
  labs(x = "Calificación", y = "Número de alumnos") +
  theme(axis.text.x = element_text(angle = 90))

"Dado un vector v de caracteres y otro vector w numérico, intercale de forma
eficiente ambos vectores. Asuma que ambos vectores tienen el mismo tamaño.
Ej: v=[“hola”, “mundo”], w=[1,6], res=[“hola”,”1”,”mundo”,”6”]."

v <- c("hola", "mundo")
w <- c(4, 6)

c(rbind(v, as.character(w)))

"Dado un data frame df con columnas: receta (tipo character), ingrediente (tipo
character), cantidad (tipo numeric), unidad (tipo character) y es_liquido
(tipo logical) se pide que, usando funciones del paquete tidyverse:
  A. Calcule la cantidad total de ingredientes líquidos (supondremos que todos
  usan la misma unidad) por receta, pero sólo de aquellas recetas que tengan un
  mínimo de 5 ingredientes (sean líquidos o no).
  B. Muestre el nombre de las 5 recetas que utilizan más ingredientes (en
  número, no en cantidad) que la media."

df <- data.frame(
  receta = c(
    # Sopa de verduras (5 ingredientes)
    "Sopa de verduras",
    # Zanahoria
    "Sopa de verduras",
    # Cebolla
    "Sopa de verduras",
    # Papa
    "Sopa de verduras",
    # Apio
    "Sopa de verduras",
    # Agua
    
    # Tarta de manzana (6 ingredientes)
    "Tarta de manzana",
    # Manzana
    "Tarta de manzana",
    # Harina
    "Tarta de manzana",
    # Huevos
    "Tarta de manzana",
    # Azúcar
    "Tarta de manzana",
    # Mantequilla
    "Tarta de manzana",
    # Canela
    
    # Resto de recetas con menos ingredientes
    "Ensalada César",
    # Lechuga
    "Ensalada César",
    # Queso
    
    "Guiso de carne",
    # Carne
    "Guiso de carne",
    # Cebolla
    "Guiso de carne",
    # Agua
    
    "Pasta al pesto",
    # Pasta
    "Pasta al pesto",
    # Albahaca
    "Pasta al pesto",
    # Aceite
    
    "Smoothie de frutas" # Leche
  ),
  ingrediente = c(
    # Sopa de verduras
    "Zanahoria",
    "Cebolla",
    "Papa",
    "Apio",
    "Agua",
    
    # Tarta de manzana
    "Manzana",
    "Harina",
    "Huevos",
    "Azúcar",
    "Mantequilla",
    "Canela",
    
    # Resto de ingredientes
    "Lechuga",
    "Queso",
    
    "Carne",
    "Cebolla",
    "Agua",
    
    "Pasta",
    "Albahaca",
    "Aceite",
    
    "Leche"
  ),
  cantidad = c(
    200,
    # Zanahoria
    100,
    # Cebolla
    300,
    # Papa
    150,
    # Apio
    1000,
    # Agua
    
    500,
    # Manzana
    200,
    # Harina
    2,
    # Huevos
    150,
    # Azúcar
    100,
    # Mantequilla
    5,
    # Canela
    
    150,
    # Lechuga
    50,
    # Queso
    
    300,
    # Carne
    200,
    # Cebolla
    500,
    # Agua
    
    400,
    # Pasta
    30,
    # Albahaca
    50,
    # Aceite
    
    500  # Leche
  ),
  unidad = c(
    "gramos",
    # Zanahoria
    "gramos",
    # Cebolla
    "gramos",
    # Papa
    "gramos",
    # Apio
    "mililitros",
    # Agua
    
    "gramos",
    # Manzana
    "gramos",
    # Harina
    "unidades",
    # Huevos
    "gramos",
    # Azúcar
    "gramos",
    # Mantequilla
    "gramos",
    # Canela
    
    "gramos",
    # Lechuga
    "gramos",
    # Queso
    
    "gramos",
    # Carne
    "gramos",
    # Cebolla
    "mililitros",
    # Agua
    
    "gramos",
    # Pasta
    "gramos",
    # Albahaca
    "mililitros",
    # Aceite
    
    "mililitros" # Leche
  ),
  es_liquido = c(
    FALSE,
    # Zanahoria
    FALSE,
    # Cebolla
    FALSE,
    # Papa
    FALSE,
    # Apio
    TRUE,
    # Agua
    
    FALSE,
    # Manzana
    FALSE,
    # Harina
    FALSE,
    # Huevos
    FALSE,
    # Azúcar
    FALSE,
    # Mantequilla
    FALSE,
    # Canela
    
    FALSE,
    # Lechuga
    FALSE,
    # Queso
    
    FALSE,
    # Carne
    FALSE,
    # Cebolla
    TRUE,
    # Agua
    
    FALSE,
    # Pasta
    FALSE,
    # Albahaca
    TRUE,
    # Aceite
    
    TRUE   # Leche
  )
)

df %>%
  group_by(receta) %>%
  filter(n() >= 5) %>%
  summarise(Cantidad = sum(es_liquido))

df %>%
  group_by(receta) %>%
  summarise(num_ingredientes = n()) %>%
  filter(num_ingredientes > mean(num_ingredientes)) %>%
  top_n(num_ingredientes, n = 5)

"Dado el mismo data frame df de antes, realice un diagrama de barras que muestre
la cantidad de recetas en donde aparece cada ingrediente. Coloree de manera
diferente los ingredientes líquidos de aquellos que no los son y dibuje una
línea horizontal que marque el valor medio de recetas por ingrediente."

df_count <- df %>%
  group_by(ingrediente) %>%
  summarise(count = n())

mean_n <- mean(df_count$count)

ggplot(df, aes(x = ingrediente, fill = es_liquido)) +
  geom_bar() +
  geom_hline(yintercept = mean_n,
             linetype = "dashed",
             color = "red") +
  theme(axis.text.x = element_text(angle = 90))

"Agregue al data frame df de antes una nueva columna llamada factor_cantidad que
sea de tipo factor ordenado con niveles (poco < normal < mucho) donde poco está
en el rango [0,15), normal está en el rango [15,100) y alto está en el rango
(100,500]. Esta columna debe representar los mismos datos que la columna
cantidad pero en los tres grupos disjuntos. Utilice esta nueva columna para
escribir en un fichero “salida.csv” (separado por comas) aquellas recetas con
todos sus ingredientes en cantidad normal"

df$cantidad
factor_cantidad <- cut(
  df$cantidad,
  breaks = c(0, 15, 100, max(df$cantidad)),
  labels = c("poco", "normal", "mucho"),
  ordered_result = TRUE
)

df$factor_cantidad <- factor_cantidad

receta_re <- df %>%
  group_by(receta) %>%
  summarise(all_normal = all(factor_cantidad == "mucho")) %>%
  filter(all_normal) %>%
  pull(receta)

write_csv(df %>%
            filter(receta %in% receta_re) %>%
            select(receta, ingrediente),
          file = "salida.csv")
