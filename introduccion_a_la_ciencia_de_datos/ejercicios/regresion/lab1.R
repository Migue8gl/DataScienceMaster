require(ggplot2)
require(tidyverse)
require(dplyr)
require(tidyr)

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

"Vamos a visualizar cada variable con scatterplot (ya que son todas numéricas)
con respecto a la variable objetivo, así comprenderemos de manera visual si
existen relaciones a priori evidentes"

california_long <- california_data %>% pivot_longer(
  cols = -MedianHouseValue,
  names_to = "var",
  values_to = "value"
)

ggplot(california_long, aes(x = value, y = MedianHouseValue)) +
  geom_point(alpha = 0.4) +
  facet_wrap( ~ var, scales = "free") +
  theme_minimal()

ggplot(california_data, aes(x = MedianHouseValue)) +
  geom_histogram(bins = 40,
                 colour = "red",
                 fill = "pink") +
  theme_minimal()

"De las gráficas que obtenemos podemos ver varias cosas. Lo primero es que no
parece haber ninguna que sea directamente lineal con relación a Y (llamaré Y a
partir de ahora a la variable objetivo).

Las variables longitud y latitud parecen crear dos distribuciones parecidas, en
cuanto a que su forma parece indicar una distribución binomial. Esto puede tener
sentido, al fin y al cabo son variables que indican una posición. Depende de
donde esté la casa valdrá más.

Households, TotalBedrooms, TotalRooms y Population parecen agrupar todos los
puntos a la izquierda, por lo que un aumento de X no parece (salvo en algunos
pocos casos) que incremente el valor de la vivienda.

La variable MedianIncome es la que más se asemeja a una relación lineal."


fit1 <- california_data %>%
  lm(formula = MedianHouseValue ~ MedianIncome, data = .)

summary(fit1)

fitted_values <- predict(fit1)

ggplot(california_data, aes(x = MedianIncome, y = MedianHouseValue)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = fitted_values), color = "blue", size = 1) +  # Fitted line
  labs(title = "Median House Value vs Median Income", x = "Median Income", y = "Median House Value") +
  theme_minimal()

"El R cuadrado es de 0.47 lo que indica el 47% de la variabilidad de mi modelo
se puede explicar con MedianIncome, lo cual para empezar no está mal, pero
indica que hay otro 53% no explicado. El estadístico F es muy grande, por lo que
hay evidencia de que MedianIncome contribuye a la predicción. El RSE es alto y
por tanto nos desviamos de media bastante del valor real de Y."

"Voy a probar a usar esa variable más la de longitud y latitud, ya que ambas
deberían aportar en combinación"

fit2 <- california_data %>%
  lm(formula = MedianHouseValue ~ Longitude, data = .)

summary(fit2)

fit3 <- california_data %>%
  lm(formula = MedianHouseValue ~ Latitude, data = .)

summary(fit3)

fit4 <- california_data %>%
  lm(formula = MedianHouseValue ~ Longitude + Latitude, data = .)

summary(fit4)

"Analizamos los valores de las casa en relación a su posición geográfica."

ggplot(california_data,
       aes(x = Longitude, y = Latitude, color = MedianHouseValue)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_minimal() +
  labs(title = "Distribución geográfica de precios de viviendas", color = "Valor medio")

"Por separadas son muy malas, pero usando Longitud y Latitud se obtiene una
variabilidad explicada del 24%, además se ser estadísticamente significativa la
aportación según el estadístico F."

fit5 <- california_data %>%
  lm(formula = MedianHouseValue ~ MedianIncome + Longitude + Latitude,
     data = .)

summary(fit5)

"Con la mejor variable encontrada y usando la combinación de latitud y longitud
se obtienen resultados mejorados. Un R cuadrado de casi 60%, aunque la fuerza
del estadístico F ha bajado."

"Pruebo a ajustar con todas las variables para aplicar la estrategia backward"

fit6 <- california_data %>%
  lm(formula = MedianHouseValue ~ ., data = .)

summary(fit6)

"No hay ninguna variable con un nivel de signficancia bajo, pero sí hay una que
es más baja que el resto. Pruebo a eliminarla."

fit7 <- california_data %>%
  lm(formula = MedianHouseValue ~ . - Households, data = .)

summary(fit7)

"No se ha perdido prácticamente nada de variabilidad y el error RSE se mantiene.
El p-valor de todas las variables es bajo, así que no tendría ningún criterio
para seguir con el backwards."

"Podríamos probar con no linealidad e interacciones de las variables que mejor
resultado nos han dado y que tienen sentido que se relacionen entre sí, como la
latitud y longitud."

fit8 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome + I(MedianIncome ^ 2) + Longitude * Latitude,
    data = .
  )

summary(fit8)

"No mejora más que la combinación lineal de todas esas características."

"Vamos a plotear la relación de cada variable entre sí para investigar relaciones
entre variables y cómo podemos hacer más interacciones entre variables."

pairs(california_data[, !(names(california_data) %in% "MedianHouseValue")])

"Podemos ver ciertas relaciones de linealidad entre (obviamente) Latitude y
Longitude, también entre TotalRooms y TotalBedrooms, HouseHolds y TotalBedrooms,
etc. Hay varias que podemos probar. Vamos a hacerlo incrementalmente."

"Añadimos ratio de dormitorios entre total de habitaciones."
fit9 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms),
    data = .
  )

summary(fit9)

"Añadimos ratio de hogares por poblacion."
fit10 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms) +
      I(Households / Population),
    data = .
  )

summary(fit10)

"Añadimos algo de no linealidad con la mejor variable."
fit11 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms) +
      I(Households / Population) +
      I(MedianIncome ^ 2) +
      I(MedianIncome ^ 2 * MedianIncome),
    data = .
  )

summary(fit11)

"Mayor grado del polinomio no parece seguir mejorando."
fit12 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms) +
      I(Households / Population) +
      I(MedianIncome ^ 2) +
      I(MedianIncome ^ 2 * MedianIncome) +
      I(MedianIncome ^ 3),
    data = .
  )

summary(fit12)

"Otras interacciones no parecen hacer mejorar el R cuadrado."
fit13 <- california_data %>%
  lm(
    formula = MedianHouseValue ~ MedianIncome +
      Longitude * Latitude +
      I(TotalBedrooms / TotalRooms) +
      I(Households / Population) +
      I(MedianIncome ^ 2) +
      I(MedianIncome ^ 2 * MedianIncome) +
      I(TotalBedrooms / Households),
    data = .
  )

summary(fit13)

"En general, los mejores resultados se han obtenido usando interacciones entre
algunas de estas variables junto con una combinación de no linealidad de la
mejor variable. No pruebo más combinaciones pues considero que el resultado es
suficientemente bueno, además existe el peligro de ir sobreajustando según mis
percepciones sesgadas. Considero que la combinación de variables usadas tiene
sentido."
