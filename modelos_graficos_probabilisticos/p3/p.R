library(lattice)
library(gridExtra)
library(gRain)
library(bnlearn)
library(Rgraphviz)

"Creamos la red inicial."
dag <- empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))
dag <- set.arc(dag, from = "A", to = "E")
dag <- set.arc(dag, from = "S", to = "E")
dag <- set.arc(dag, from = "E", to = "O")
dag <- set.arc(dag, from = "E", to = "R")
dag <- set.arc(dag, from = "O", to = "T")
dag <- set.arc(dag, from = "R", to = "T")
dag
plot(dag)

"Cargamos los resultados de una encuesta."
survey <- read.table("survey.txt", header = TRUE, colClasses = "factor")
head(survey)

"Para estimar los parámetros de una red bayesiana (las probabilidades
condicionales) es posible usar el paquete bnlearn para hacerlo mediante
métodos frecuentistas y de máxima verosimilitud."
bn.mle <- bn.fit(dag, data = survey, method = "mle")

"También podrían haberse calculado las distribuciones a mano mediante otros
métodos..."

"Ahora podemos ver las distribuciones."
bn.mle

"Alternativamente es posible usar el mismo paquete para usar un método bayesiano.
Este se especifica en el parámetro method de la misma función."
bn.mle <- bn.fit(dag,
                 data = survey,
                 method = "bayes",
                 iss = 10)

"Las probabilidades a posteriori se calculan a partir de una distribución
uniforme sobre cada table de distribución de probabilidad condicional. El
argumento iss es el tamaño muestral equivalente. Es decir, al principio la
probabilidad de cada variable era uniforme (probabilidad a priori). A mayor sea
iss más cerca quedará la distribución a posterior de la uniforme, por ello suele
ser una valor bajo."

"Ahora vamos a aplicar el test de G^2 (basado en la medida de información y en
el test clásico x^2). Este test sirve para realizar tests de independencia
condicional entre dos variables, posiblemente condicionadas a otras variables."
ci.test("T", "E", c("O", "R"), test = "mi", data = survey)
ci.test("T", "E", c("O", "R"), test = "x2", data = survey)

"Los p-valores son muy significativos, por lo que el enlace E -> T no es
signficativo, o lo que es lo mismo, ambos nodos son independientes."

"Se pueden obtener todos los p-valores de cada hijo con respecto a sus padres."
arc.strength(dag, data = survey, criterion = "x2")

"El score nos permite ver como de buena es una red con respecto a un conjunto
de datos."
score(dag, data = survey, type = "bic")
score(dag,
      data = survey,
      type = "bde",
      iss = 10)

"Usando un score podemos comparar dos redes entre si y comprobar si añadir
enlaces ayuda."
dag4 <- set.arc(dag, from = "E", to = "T")
score(dag4, data = survey, type = "bic")

"Estos scores penalizan mucho la complejidad de una red, aunque mejore un poco."

"Para aprender un dag hay dos criterios (por test de independencia y por scores).
Podemos usar un algoritmo greedy basado en scores."
learned <- hc(survey)
modelstring((learned))
score(learned, data = survey, type = "bic")

"Para crear las redes con los algoritmos de restricciones se pueden hacer con
las funciones gs, iamb, fast.iamb, inter.iamb, mmpc, si.hiton.pc y
hpc."

"Para comparar dos redes se puede usar el algoritmo compare, que devuelve el ratio
de falsos positivos, verdaderos positivos, etc."