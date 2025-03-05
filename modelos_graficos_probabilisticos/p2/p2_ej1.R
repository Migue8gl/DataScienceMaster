library(bnlearn)
library(gRain)

"Creación del dag con sus nodos"
dag <- empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))
dag

"Añadido de aristas entre los nodos"
dag <- set.arc(dag, from = "A", to = "E")
dag <- set.arc(dag, from = "S", to = "E")
dag <- set.arc(dag, from = "E", to = "O")
dag <- set.arc(dag, from = "E", to = "R")
dag <- set.arc(dag, from = "O", to = "T")
dag <- set.arc(dag, from = "R", to = "T")
dag

"Modificar el modelo como string"
modelstring(dag)
dag2 <- model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")
all.equal(dag, dag2)

"Lista de nodos y lista de arista"
nodes(dag)
arcs(dag)

"Dibujar DAG"
plot(dag)

"Definición de estados de cada variable"
A.st <- c("young", "adult", "old")
S.st <- c("M", "F")
E.st <- c("high", "uni")
O.st <- c("emp", "self")
R.st <- c("small", "big")
T.st <- c("car", "train", "other")

"Definición de probabilidades"
A.prob <- array(c(0.30, 0.50, 0.20),
                dim = 3,
                dimnames = list(A = A.st))
A.prob
S.prob <- array(c(0.60, 0.40), dim = 2, dimnames = list(S = S.st))
S.prob
O.prob <- array(c(0.96, 0.04, 0.92, 0.08),
                dim = c(2, 2),
                dimnames = list(O = O.st, E = E.st))
O.prob
R.prob <- array(c(0.25, 0.75, 0.20, 0.80),
                dim = c(2, 2),
                dimnames = list(R = R.st, E = E.st))
R.prob
E.prob <- array(
  c(
    0.75,
    0.25,
    0.72,
    0.28,
    0.88,
    0.12,
    0.64,
    0.36,
    0.70,
    0.30,
    0.90,
    0.10
  ),
  dim = c(2, 3, 2),
  dimnames = list(E = E.st, A = A.st, S = S.st)
)
E.prob
T.prob <- array(
  c(
    0.48,
    0.42,
    0.10,
    0.56,
    0.36,
    0.08,
    0.58,
    0.24,
    0.18,
    0.70,
    0.21,
    0.09
  ),
  dim = c(3, 2, 2),
  dimnames = list(T = T.st, O = O.st, R = R.st)
)
T.prob

"Creación de la red bayesiana"
cpt <- list(
  A = A.prob,
  S = S.prob,
  E = E.prob,
  O = O.prob,
  R = R.prob,
  T = T.prob
)

bn <- custom.fit(dag, cpt)
nparams(bn)
bn

"Dependencia entre variables (d-separación) y caminos"
dsep(dag, x = "S", y = "R")
path.exists(dag, from = "S", to = "R")

"Dependencia dada una evidencia (dependencia condicional)"
dsep(dag,
     x = "S",
     y = "T",
     z = c("O", "R"))

"Para el algoritmo de inferencia"
junction <- compile(as.grain(bn))

"Distribución de probabilidad a posteriori para una variable dada una evidencia"
querygrain(junction, nodes = "T")$T

"Definir evidencias"
jsex <- setEvidence(junction, nodes = "S", states = "F")
querygrain(jsex, nodes = "T")$T


jres <- setEvidence(junction, nodes = "R", states = "small")
querygrain(jres, nodes = "T")$T

"Probabilidad conjunta condicional dada la evidencia para más de una variable"
jedu <- setEvidence(junction, nodes = "E", states = "high")
querygrain(jedu, nodes = c("S", "T"), type = "joint")

"Hay varios tipos de consultas, también se puede obtener la distribución de
probabilidad marginal de cada variable o incluso la distribución de probabilidad
condicional de la primera variable en nodes condicionada al resto de variables (y
a la evidencia)."
querygrain(jedu, nodes = c("S", "T"), type = "marginal")
querygrain(jedu, nodes = c("S", "T"), type = "conditional")

"Inferencia aproximada"
cpquery(bn,
        event = (S == "M") & (T == "car"),
        evidence = (E == "high")) # Muestreo lógico
cpquery(
  bn,
  event = (S == "M") & (T == "car"),
  evidence = (E == "high"),
  n = 10 ^ 6
) # Muestreo lógico con mayor número de muestras aleatorias

cpquery(
  bn,
  event = (S == "M") &
    (T == "car"),
  evidence = list(E = "high"),
  method = "lw"
) # Ponderación por similitud
