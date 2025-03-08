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
cptA <- matrix(c(0.1, 0.9), ncol = 2, dimnames = list(NULL, A.st))
cptT <- matrix(c(0.05, 0.95, 0.01, 0.99),
               ncol = 2,
               dimnames = list(A.st, T.st))
cptS <- matrix(c(0.5, 0.5), ncol = 2, dimnames = list(NULL, S.st))
cptL <- matrix(c(0.1, 0.9, 0.01, 0.99),
               ncol = 2,
               dimnames = list(S.st, L.st))
cptB <- matrix(c(0.6, 0.4, 0.3, 0.7),
               ncol = 2,
               dimnames = list(S.st, B.st))
cptE <- array(c(1, 0, 1, 0, 1, 0, 0, 1),
              dim = c(2, 2, 2),
              dimnames = list(E.st, L.st, T.st))
cptX <- matrix(c(0.98, 0.02, 0.05, 0.95),
               ncol = 2,
               dimnames = list(E.st, X.st))
cptD <- array(
  c(0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.1, 0.9),
  dim = c(2, 2, 2),
  dimnames = list(D.st, E.st, B.st)
)

"Estructura de la red bayesiana"
net <- custom.fit(dag,
                  dist = list(
                    A = cptA,
                    T = cptT,
                    S = cptS,
                    L = cptL,
                    B = cptB,
                    E = cptE,
                    X = cptX,
                    D = cptD
                  ))

net
junction <- compile(as.grain(net))

"1. Tablas de probabilidad marginal de padecer tuberculosis, de paceder
cáncer de pulmón y de padecer bronquitis. Usa el método exacto y
los dos aproximados."
# Exacto
querygrain(junction, nodes = c("T", "L", "B"), type = "marginal")

# Aprox
cpquery(net,
        event = (T == "yes"),
        evidence = TRUE,
        method = "lw")
cpquery(net,
        event = (L == "yes"),
        evidence = TRUE,
        method = "lw")
cpquery(net,
        event = (B == "yes"),
        evidence = TRUE,
        method = "lw")

cpquery(net, event = (T == "yes"), evidence = TRUE)
cpquery(net, event = (L == "yes"), evidence = TRUE)
cpquery(net, event = (B == "yes"), evidence = TRUE)


"2. Tablas de probabilidad marginal a posteriori de padecer cada una de
las tres enfermedades anteriores dado que se sabe que el paciente
visitó Asia. Usa el método exacto y los dos aproximados."
# Exacto
junction_A_yes <- setEvidence(junction, nodes = "A", states = "yes")
querygrain(junction_A_yes,
           nodes = c("T", "L", "B"),
           type = "marginal")

# Aprox
cpquery(
  net,
  event = (T == "yes"),
  evidence = list(A = "yes"),
  method = "lw"
)
cpquery(
  net,
  event = (L == "yes"),
  evidence = list(A = "yes"),
  method = "lw"
)
cpquery(
  net,
  event = (B == "yes"),
  evidence = list(A = "yes"),
  method = "lw"
)

cpquery(net, event = (T == "yes"), evidence = (A == "yes"))
cpquery(net, event = (L == "yes"), evidence = (A == "yes"))
cpquery(net, event = (B == "yes"), evidence = (A == "yes"))

"3. Tablas de probabilidad marginal a posteriori de padecer cada una de
las tres enfermedades anteriores dado que se sabe que el paciente
visitó Asia, y tiene asma. Usa el método exacto y los dos aproxima-
dos."
# Exacto
junction_A_and_D_yes <- setEvidence(junction,
                                    nodes = c("A", "D"),
                                    states = list(A = "yes", D = "yes"))
querygrain(junction_A_yes,
           nodes = c("T", "L", "B"),
           type = "marginal")

# Aprox
cpquery(
  net,
  event = (T == "yes"),
  evidence = list(A = "yes", D = "yes"),
  method = "lw"
)
cpquery(
  net,
  event = (L == "yes"),
  evidence = list(A = "yes", D = "yes"),
  method = "lw"
)
cpquery(
  net,
  event = (B == "yes"),
  evidence = list(A = "yes", D = "yes"),
  method = "lw"
)

cpquery(net,
        event = (T == "yes"),
        evidence = (A == "yes" & D == "yes"))
cpquery(net,
        event = (L == "yes"),
        evidence = (A == "yes" & D == "yes"))
cpquery(net,
        event = (B == "yes"),
        evidence = (A == "yes" & D == "yes"))

"4. Tabla de probabilidad conjunta de padecer cáncer de pulmón y bron-
quitis. Indica cual es la probabilidad de la configuración más proba-
ble para las variables de estas dos enfermadades."
querygrain(junction, nodes = c("L", "B"), type = "joint")

# La configuración más probable es la de no tener ni cancer ni bronquitis

"5. Tabla de probabilidad conjunta de padecer cáncer de pulmón y bron-
quitis dado que el paciente no visitó Asia y el paciente es fumador.
Indica cual es la probabilidad de la configuración más probable pa-
ra las variables de estas dos enfermadades cuando el paciente no
visitó Asia y el paciente es fumador."
junction_A_no_and_S_yes <- setEvidence(junction,
                                       nodes = c("A", "S"),
                                       states = list(A = "no", S = "yes"))
querygrain(junction_A_no_and_S_yes,
           nodes = c("L", "B"),
           type = "joint")

# Es mucho más probable que no tenga cáncer de pulmón, pero que sí tenga
# bronquitis dado que no visitó Asia y que si fuma

"6. Si no se conoce el valor de ninguna variable, ¿qué comando(s) pode-
mos usar para saber si visitar Asia influye en tener cáncer de pulmón
o no?."
querygrain(junction, nodes = c("A", "L"), type = "conditional")

# Visitar Asia no afecta prácticamente nada en tener o no cancer