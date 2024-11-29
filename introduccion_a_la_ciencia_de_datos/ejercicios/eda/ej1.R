library(tidyverse)

# Leer el archivo CSV
data <- read.csv("USAirlines.csv", header = TRUE, sep = ",")

# Verificamos dimensión y miramos el nombre de las columnas
dim(data)
colnames(data)
str(data)

# Calculamos la media de cada columna numérica
# Quitamos nulos y en las columnas (margin == 2)
mean_values <- apply(data, 2, mean, na.rm = TRUE)
mean_values

# Calculamos la mediana de cada columna numérica
median_values <- apply(data, 2, median, na.rm = TRUE)
median_values

# Sapply aplica range a cada columna del dataset
# Range devuelve un vector de dos elementos, el mínimo y el máximo
rangos <- sapply(data, range, na.rm = TRUE)
rangos

# Calculamos varianza y desviación estandar en la columna price, por ejemplo
data %>% 
  summarise(
    price_var = var(price, na.rm = TRUE),
    price_std_dev = sd(price, na.rm = TRUE)
  )

ggplot(data, aes(x=cost, y=output))
