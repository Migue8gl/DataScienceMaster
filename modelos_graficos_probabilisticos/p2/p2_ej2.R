library(bnlearn)
library(gRain)

"Creación del dag con sus nodos"
dag <- model2network("[A][T|A][S][L|S][B|S][E|T:L][X|E][D|E:B]")

"Lista de nodos y lista de arista"
nodes(dag)
arcs(dag)

"Dibujar DAG"
plot(dag)

"Definición de estados de cada variable"
A.st <- c("yes", "no")  # Visitó Asia
T.st <- c("yes", "no")  # Tuberculosis
S.st <- c("yes", "no")  # Es fumador
L.st <- c("yes", "no")  # Cáncer de pulmón
B.st <- c("yes", "no")  # Bronquitis
E.st <- c("yes", "no")  # Tuberculosis o cáncer de pulmón
X.st <- c("yes", "no")  # Rayos X positivo
D.st <- c("yes", "no")  # Disnea

"Definición de probabilidades"
