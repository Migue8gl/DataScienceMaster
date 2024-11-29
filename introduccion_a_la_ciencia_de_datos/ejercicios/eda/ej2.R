# Instalación y carga de librerías
if (!require(ISLR)) install.packages("ISLR")
if (!require(ggplot2)) install.packages("ggplot2")
library(ISLR)
library(ggplot2)

# Cargar el dataset Carseats
data(Carseats)

# 1. Encontrar variables con skewness
calculate_skewness <- function(x) {
  n <- length(x)
  (sum((x - mean(x))^3) / n) / (sum((x - mean(x))^2) / n)^(3/2)
}

skewness_values <- sapply(Carseats[sapply(Carseats, is.numeric)], calculate_skewness)
print(skewness_values)

# 2. Generar listas de variables con skewness a la derecha y a la izquierda
right_skewed <- names(skewness_values[skewness_values > 0])
left_skewed <- names(skewness_values[skewness_values < 0])

print("Variables con skewness a la derecha:")
print(right_skewed)
print("Variables con skewness a la izquierda:")
print(left_skewed)

# 3. Averiguar qué variables no están distribuidas de forma normal
shapiro_test <- function(x) {
  test <- shapiro.test(x)
  return(test$p.value)
}

normality_test <- sapply(Carseats[sapply(Carseats, is.numeric)], shapiro_test)
non_normal_vars <- names(normality_test[normality_test < 0.05])

print("Variables que no siguen una distribución normal:")
print(non_normal_vars)

# Crear gráficos Q-Q para variables no normales
for (var in non_normal_vars) {
  p <- ggplot(Carseats, aes_string(sample = var)) +
    stat_qq() +
    stat_qq_line() +
    ggtitle(paste("Q-Q plot for", var))
  print(p)
}

# 4. Encontrar correlaciones y crear un gráfico de correlación
correlation_matrix <- cor(Carseats[sapply(Carseats, is.numeric)])
if (!require(corrplot)) install.packages("corrplot")
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
