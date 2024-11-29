require(ggplot2)
require(tidyverse)
require(kknn)

"Leemos el dataset de california. En este proceso nos saltamos las líneas que
contengan metadatos y añadimos las columnas a mano"

california_data <- read.table(
  "california.dat",
  sep = ",",
  skip = 13,
  col.names = c(
    "Longitude",
    "Latitude",
    "HousingMedianAge",
    "TotalRooms",
    "TotalBedrooms",
    "Population",
    "Households",
    "MedianIncome",
    "MedianHouseValue"
  )
)

"Obtenemos resumen estadístico de cada variable en el dataset y vemos el tipo de
cada variable"

summary(california_data)
str(california_data)

"Obtenemos el primer modelo usando knn con el dataset de California. Dejamos los
hiperparámetros por defecto"

fitknn1 <- kknn(MedianHouseValue ~ ., california_data, california_data)
names(fitknn1)

"Visualizamos los datos originales vs los puntos ajustados del knn"
ggplot(data = california_data, aes(x = MedianIncome, y = MedianHouseValue)) +
  geom_point() +
  geom_point(aes(y = fitknn1$fitted.values),
             color = "blue",
             shape = 20) +
  labs(x = "Median Income", y = "Median House Value", title = "Median House Value vs Median Income con valores ajustados por kNN") +
  theme_minimal()

"Calculamos el RMSE"
yprime <- fitknn1$fitted.values
sqrt(sum((
  california_data$MedianHouseValue - yprime
) ^ 2) / length(yprime))

"El mejor modelo obtenido con regresión lineal en el pasado laboratorio obtenia
el doble de RMSE, lo que ocurre es que era más interpretabe, mientras que
knn obtiene mejores resultados a cambio de serlo menos."

"Vamos a probar algunas combinaciones igual que se hizo con regresión lineal. Lo
probamos para la mejor versión que encontré en regresión lineal."

fitknn2 <- kknn(
  MedianHouseValue ~ MedianIncome +
    Longitude * Latitude +
    I(TotalBedrooms / TotalRooms) +
    I(Households / Population) +
    I(MedianIncome ^ 2) +
    I(MedianIncome ^ 3),
  california_data,
  california_data
)
yprime <- fitknn2$fitted.values
sqrt(sum((
  california_data$MedianHouseValue - yprime
) ^ 2) / length(yprime))

"Mejora, pero no mucho considerando el número tan grande que es. Lo mejor para
un algoritmo no tiene por qué ser lo mejor en otro."

fitknn3 <- kknn(
  MedianHouseValue ~ MedianIncome +
    Longitude * Latitude ,
  california_data,
  california_data
)
yprime <- fitknn3$fitted.values
sqrt(sum((
  california_data$MedianHouseValue - yprime
) ^ 2) / length(yprime))

"Teniendo en cuenta solo latitud y longitud (además de la variable con principal
relación) funciona mejor que con las transformaciones realizadas previamente."

fitknn4 <- kknn(MedianHouseValue ~ . - HousingMedianAge,
                california_data,
                california_data)
yprime <- fitknn4$fitted.values
sqrt(sum((
  california_data$MedianHouseValue - yprime
) ^ 2) / length(yprime))

ggplot(data = california_data, aes(x = MedianIncome, y = MedianHouseValue)) +
  geom_point() +
  geom_point(aes(y = fitknn2$fitted.values),
             color = "red",
             shape = 20) +
  labs(x = "Median Income", y = "Median House Value", title = "Median House Value vs Median Income con valores ajustados por kNN") +
  theme_minimal()

ggplot(data = california_data, aes(x = MedianIncome, y = MedianHouseValue)) +
  geom_point() +
  geom_point(aes(y = fitknn3$fitted.values),
             color = "orange",
             shape = 20) +
  labs(x = "Median Income", y = "Median House Value", title = "Median House Value vs Median Income con valores ajustados por kNN") +
  theme_minimal()

ggplot(data = california_data, aes(x = MedianIncome, y = MedianHouseValue)) +
  geom_point() +
  geom_point(aes(y = fitknn4$fitted.values),
             color = "green",
             shape = 20) +
  labs(x = "Median Income", y = "Median House Value", title = "Median House Value vs Median Income con valores ajustados por kNN") +
  theme_minimal()

"Vamos a hacer 5-fold-cross validation con particiones de test para ver la
calidad de los algoritmos. Como ejemplo ponemos el LM con todos los datos
para california."

nombre <- "california"
run_lm_fold <- function(i, x, tt = "test") {
  file <- paste(x, "-5-", i, "tra.dat", sep = "")
  x_tra <- read.csv(file, comment.char = "@", header = FALSE)
  file <- paste(x, "-5-", i, "tst.dat", sep = "")
  x_tst <- read.csv(file, comment.char = "@", header = FALSE)
  In <- length(names(x_tra)) - 1
  names(x_tra)[1:In] <- paste ("X", 1:In, sep = "")
  names(x_tra)[In + 1] <- "Y"
  names(x_tst)[1:In] <- paste ("X", 1:In, sep = "")
  names(x_tst)[In + 1] <- "Y"
  if (tt == "train") {
    test <- x_tra
  }
  else {
    test <- x_tst
  }
  fitMulti = lm(Y ~ ., x_tra)
  yprime = predict(fitMulti, test)
  sum(abs(test$Y - yprime) ^ 2) / length(yprime) ##MSE
}
lmMSEtrain <- mean(sapply(1:5, run_lm_fold, nombre, "train"))
lmMSEtest <- mean(sapply(1:5, run_lm_fold, nombre, "test"))

"Ahora vamos a hacerlo con el modelo que destilamos en el primer laboratorio."

run_lm_fold <- function(i, x, tt = "test") {
  file_tra <- paste(x, "-5-", i, "tra.dat", sep = "")
  x_tra <- read.csv(file_tra, comment.char = "@", header = FALSE)
  
  file_tst <- paste(x, "-5-", i, "tst.dat", sep = "")
  x_tst <- read.csv(file_tst, comment.char = "@", header = FALSE)
  
  In <- length(names(x_tra)) - 1
  names(x_tra)[1:In] <- c(
    "Longitude",
    "Latitude",
    "HousingMedianAge",
    "TotalRooms",
    "TotalBedrooms",
    "Population",
    "Households",
    "MedianIncome"
  )
  names(x_tra)[In + 1] <- "MedianHouseValue"
  names(x_tst)[1:In] <- c(
    "Longitude",
    "Latitude",
    "HousingMedianAge",
    "TotalRooms",
    "TotalBedrooms",
    "Population",
    "Households",
    "MedianIncome"
  )
  names(x_tst)[In + 1] <- "MedianHouseValue"
  
  test <- if (tt == "train")
    x_tra
  else
    x_tst
  
  fitMulti <- lm(
    MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms) +
      I(Households / Population) +
      I(MedianIncome ^ 2) +
      I(MedianIncome ^ 3),
    data = x_tra
  )
  
  yprime <- predict(fitMulti, test)
  mse <- sum((test$MedianHouseValue - yprime) ^ 2) / length(yprime)
  return(mse)
}

my_lmMSEtrain <- mean(sapply(1:5, run_lm_fold, nombre, "train"))
my_lmMSEtest <- mean(sapply(1:5, run_lm_fold, nombre, "test"))

"El error no es mucho mejor que el conseguido por todas las variables, pero aún
así mejora y el test es razonable y por debajo del error de train, por lo que
parece que el modelo generaliza bien. Ahora vamos a obtener el RSME de knn."

run_lm_fold <- function(i, x, tt = "test") {
  file_tra <- paste(x, "-5-", i, "tra.dat", sep = "")
  x_tra <- read.csv(file_tra, comment.char = "@", header = FALSE)
  
  file_tst <- paste(x, "-5-", i, "tst.dat", sep = "")
  x_tst <- read.csv(file_tst, comment.char = "@", header = FALSE)
  
  In <- length(names(x_tra)) - 1
  names(x_tra)[In + 1] <- "MedianHouseValue"
  names(x_tst)[In + 1] <- "MedianHouseValue"
  
  test <- if (tt == "train")
    x_tra
  else
    x_tst
  
  fitknn <- kknn(MedianHouseValue ~ ., x_tra, test)
  
  yprime <- fitknn$fitted.value
  mse <- sum((test$MedianHouseValue - yprime) ^ 2) / length(yprime)
  return(mse)
}

knnMSEtrain <- mean(sapply(1:5, run_lm_fold, nombre, "train"))
knnMSEtest <- mean(sapply(1:5, run_lm_fold, nombre, "test"))

"Los resultados con KNN son bastante mejores a priori. El test tiene un error 
un poco alejado del train, pero sigue siendo mejor que el LM."
