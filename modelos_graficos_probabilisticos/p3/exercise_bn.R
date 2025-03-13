library(lattice)
library(gridExtra)
library(gRain)
library(bnlearn)
library(Rgraphviz)

# Read the data and plot the dag structure
sachs <- readRDS("sachs")
dag <- bn.net(sachs)
graphviz.plot(dag)

# Simulate data set
# 200 size
data_200 <- rbn(sachs, n = 200)
# 500 size
data_5000 <- rbn(sachs, n = 5000)
head(data_200)
head(data_5000)

# Now we have to learn the structure with different methods

# Score-based
learned_score_200 <- hc(data_200)
modelstring(learned_score_200)
graphviz.plot(learned_score_200)

learned_score_5000 <- hc(data_5000)
modelstring(learned_score_5000)
graphviz.plot(learned_score_5000)

# Independence-test-based
learned_independence_200 <- iamb(data_200)
learned_independence_200 <- cextend(learned_independence_200)
modelstring(learned_independence_200)
graphviz.plot(learned_independence_200)

learned_independence_5000 <- iamb(data_5000)
learned_independence_5000 <- cextend(learned_independence_5000)
modelstring(learned_independence_5000)
graphviz.plot(learned_independence_5000)

# Learn parameters of the DAGs
fit_hc_200 <- bn.fit(learned_score_200, data_200)
fit_hc_5000 <- bn.fit(learned_score_5000, data_5000)
fit_iamb_200 <- bn.fit(learned_independence_200, data_200)
fit_iamb_5000 <- bn.fit(learned_independence_5000, data_5000)

# Originales
sachs$Erk
sachs$PKA

# HC
fit_hc_200$Erk
fit_hc_5000$Erk
fit_hc_200$PKA
fit_hc_5000$PKA

# IAMB
fit_iamb_200$Erk
fit_iamb_5000$Erk
fit_iamb_200$PKA
fit_hc_5000$PKA
