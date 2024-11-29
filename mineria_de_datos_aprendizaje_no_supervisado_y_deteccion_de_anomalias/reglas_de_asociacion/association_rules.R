library(mlbench)
library(tidyverse)
library(arules)
data(Zoo)

# All logical to factor
# Divide into has legs and has no legs
Zoo <- Zoo %>% mutate(no_legs = legs == 0,
                      has_legs = legs > 0,
                      legs = NULL) %>% mutate_if(is.logical, as.factor)

# View dataset variables and visualization
Zoo[1:2, ]

Zoo %>% filter(airborne == TRUE & hair == TRUE)


zoo_t <- as(Zoo, "transactions")
zoo_t
summary(zoo_t)
image(zoo_t)
itemFrequencyPlot(zoo_t, support = 0.1, cex.names = 0.8)

i_zoo_t <- apriori(zoo_t, parameter = list(
  support = 0.1,
  confidence = 0.8,
  minlen = 2
))
i_zoo_t <- sort(i_zoo_t, by = "support")
inspect(head(i_zoo_t, n = 10))
size(i_zoo_t)
barplot(table(size(i_zoo_t)), xlab = "itemset size", ylab = "count")
inspect(i_zoo_t[size(i_zoo_t) == 1])

rules <- apriori(zoo_t, parameter = list(
  support = 0.1,
  confidence = 0.8,
  minlen = 2
))
inspect(head(rules, 30))

rulesSorted = sort(rules, decreasing = FALSE, by = "confidence")
inspect(head(rulesSorted, 15))

redundant <- is.redundant(x = rulesSorted, measure = "confidence")
rulesPruned <- rulesSorted[!redundant]

inspect(head(rulesPruned, 80))
