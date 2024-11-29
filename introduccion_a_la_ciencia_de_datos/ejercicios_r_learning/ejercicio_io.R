"Pida al usuario que introduzca con el teclado una cadena de caracteres s y un
número n e imprima en pantalla n veces la cadena s (sin espacios entre palabras)
tal como se ve en el ejemplo (notar que no hay un [1] al principio...)"

s <- readline("Introduzca una cadena de caracteres: ")
n <- readline("Introduzca un número: ")
strrep(s, n)

"Crea tres ficheros llamados dos.txt, tres.txt y cinco.txt que contengan la
tabla de 2, 3 y 5 respectivamente (sólo incluye los 10 primeros valores de cada
uno, un número en una línea separada, SOLO el número, nada más)."

c_base <- 1:10
tabla_dos <- c_base * 2
tabla_tres <- c_base * 3
tabla_cinco <- c_base * 5

write.table(tabla_dos,
            "dos.txt",
            row.names = FALSE,
            col.names = FALSE)
write.table(tabla_tres,
            "tres.txt",
            row.names = FALSE,
            col.names = FALSE)
write.table(tabla_cinco,
            "cinco.txt",
            row.names = FALSE,
            col.names = FALSE)

"Escribe las cinco primeras filas de la matriz creada en el último ejercicio en
un nuevo fichero llamado prime.txt y las cinco últimas en otro fichero llamado
fin.txt. Ambos ficheros deben tener los datos separados por comas."

matriz <- matrix(c(tabla_dos, tabla_tres, tabla_cinco),
                 ncol = 3,
                 byrow = FALSE)
write.table(matriz[1:5, ], sep = ",", file = "prime.txt")

write.table(matriz[as.integer(nrow(matriz) - 4):as.integer(nrow(matriz)), ], file = "fin.txt", sep =
              ",")

"Dados dos números, f y c (dados por el usuario mediante el teclado), cree una
figura cuadrada de f filas y c columnas con el carácter \"x\" (sin espacios).
Vea a continuación un ejemplo para f=4 y c=3 (notar que no hay espacios en
blanco ni [1,] ni cosas raras...):"

f <- readline("Introduzca un número: ")
c <- readline("Introduzca un número: ")

fila <- paste(rep("x", c), collapse = "")
figura <- paste(rep(fila, f), collapse = "\n")

cat(figura)

"Cargue la primer y tercera hojas del fichero resultados.xls y muestre un
gráfico que compare, para los dos datasets, el resultado en entrenamiento y
test a medida que aumenta la cantidad de bits utilizados."

library("readxl")
library(tidyverse)
data1 <- read_excel("results.xlsx",
                    col_names = FALSE,
                    sheet = 1,
                    skip = 2)
data2 <- read_excel("results.xlsx",
                    sheet = 3,
                    col_names = FALSE,
                    skip = 2)
colnames(data1)[1:6] <- c("dataset", "model", "bits", "nodes", "train", "test")
colnames(data2)[1:6] <- c("dataset", "model", "bits", "nodes", "train", "test")

data1$source <- "sheet1"
data2$source <- "sheet3"
combined_data <- bind_rows(data1, data2)

ggplot(combined_data, aes(x = bits)) +
  geom_line(aes(
    y = train,
    color = "train",
    linetype = source
  )) +
  geom_line(aes(
    y = test,
    color = "test",
    linetype = source
  )) +
  scale_color_manual(values = c("train" = "blue", "test" = "orange")) +
  theme_classic() +
  labs(title = "Accuracy over bits for Sheet1 and Sheet3", x = "Bits", y = "Accuracy") +
  facet_wrap( ~ model) +
  theme(legend.position = "top")
