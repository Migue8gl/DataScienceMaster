library(tidyverse)
library(moments)
library("corrplot")
library(ggfortify)
library(MVN)

"Leemos los datos saltandonos las cabeceras iniciales y después las parseamos
a mano. Seguido, vamos a ver unos pocos datos con head para hacernos una primera
idea."

data <- read.csv("wankara/wankara.dat", skip = 14, header = FALSE)
colnames(data) <- c(
  "Max_temperature",
  "Min_temperature",
  "Dewpoint",
  "Precipitation",
  "Sea_level_pressure",
  "Standard_pressure",
  "Visibility",
  "Wind_speed",
  "Max_wind_speed",
  "Mean_temperature"
)
head(data)

"Con la función de summary obtenemos un resumen estadístico general de cada
columna. Además vamos a obtener otros datos informativos como la dimensión,
y la estructura."

summary(data)
dim(data)
str(data)
colSums(is.na(data))

"Los datos parecen bastante buenos a priori. No hay escalas numéricas muy
grandes, no hay datos faltantes y en principio parece que las distribuciones
se centran en la media (a excepción de algunas variables como Precipitation y
alguna más). En principio lo más intuitivo sería analizar visualmente las
variables."

plot_distribution <- function(data,
                              var_name,
                              binwidth = 0.1,
                              fill_color = "pink") {
  ggplot(data, aes_string(x = var_name)) +
    geom_density(fill = fill_color,
                 color = "black",
                 alpha = 0.7) +
    labs(
      title = paste("Distribution of", var_name),
      x = var_name,
      y = "Frequency"
    ) +
    theme_minimal()
}

for (column in names(data)) {
  p <- plot_distribution(data, column)
  print(p)
}

"Las variables de temperatura máxima y mínima parecen adecuarse a la forma de
una pseudo-bimodal (Tienen dos picos y una depresión claramente diferenciados).
La variable de precipitación es asimétrica extrema, con valores prácticamente
nulos (entiendase por nulo el cero). Las relativas a la presión son
distribuciones que si bien no son normales exactas, se asimilan mucho (ambas
poseen la peculiaridad de un pequeño \"bulto\"para presiones de entre 26-30. En
cuanto al resto, hay variedad de distribuciones con asimetrías."

"La bimodalidad en las temperaturas puede darse por varios motivos, pero el que
considero más probable es la toma de datos en distintas zonas geográficas de
Ankara, dando lugar a distintas modas. El calentamiento climático podría ser
factible, pero no sería tan evidente y se necesitarían tomas de muchos años."

"Vamos a escalar los datos, ya que así se reduciran las diferencias entre
variables. Escalar permite que todas las variables contribuyan equitativamente a
la distancia calculada, evitando que una variable desproporcionada influya en
la detección de outliers."

data <- as.data.frame(scale(data))
summary(data)

"A continuación vamos a aplicar test de normalidad a aquellas variables más
prometedoras y ver si la cumplen. Además obtenemos los qqplots de todas las
variables para ver como de cerca están las distribuciones de la normal teórica."

long_data <- data %>%
  pivot_longer(
    cols = setdiff(names(data), "Mean_temperature"),
    names_to = "Variable",
    values_to = "Value"
  )

ggplot(long_data, aes(sample = Value)) +
  stat_qq() +
  stat_qq_line() +
  facet_wrap(~ Variable) +
  labs(title = "Qqplot of every variable", y = "Variable quantiles", x = "Theoretical quantiles") +
  theme_minimal()

shapiro.test(data$Sea_level_pressure)
shapiro.test(data$Standard_pressure)

"Vamos a analizar los boxplots de las variables"

ggplot(long_data, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  labs(title = "Boxplots", x = "Variable", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

"Observando las variables y sus valore máximos y mínimos no consideraría que
existan outliers univariantes, ya que son valores posibles y nada irreales.
Las precipitaciones es la que tiene valores más extremos, pero por lo general,
al ser una zona seca, es normal que la mayoría de valores sean cero. Más tarde
comprobaré si pueden existir outliers multivariantes."

"Si bien es cierto que el test rechaza (no es una distribución normal). Los
qqplot de las variables concernientes a la presión son muy buenos. De hecho en
las gráficas de densidad hemos observado que no presentan formas anómalas. Dados
estos resultados, pese a no pasar los test, podríamos tomar una postura un poco
más relajada en cuanto a restricciones de la normal en lo que se refiere a esas dos
variables, sobre todo en algunas técnicas (si se llegan a usar) como ANOVA, que
son muy robustas a violaciones de normalidad.
El resto de variables son también bastante normales (algo asimétricas).
Se dan excepciones, por ejemplo, La visibilidad tiene una cola muy larga a la
derecha, al igual que la máxima velocidad del viento (aunque ya hemos visto que
las variables de viento parecen bimodales)."

"Vamos a ver la correlación de las características"

cor_matrix <- cor(data)
corrplot(cor_matrix,
         method = "number",
         type = "upper",
         tl.col = "black")

"Existen fuertes coeficientes de correlación entre variables de mínimo, máximo y
temperatura media. Estas correlaciones son muy grandes y positivas, cuando crece
una variable lo hace la variable objetivo. Además el punto de condensación o
dewpoint también tiene correlación con la variable objetivo. Estas correlaciones
tienen sentido. El hecho de que el punto de condensación o dewpoint también esté
correlacionado con la variable objetivo refuerza la idea de que las condiciones
atmosféricas, como la humedad y la temperatura, están relacionadas con el
comportamiento de la variable de interés. En particular, el dewpoint es un
indicador importante de la humedad en el aire, lo que podría influir
directamente en el fenómeno que se está modelando."

ggplot(data, aes(x = Dewpoint, y = Mean_temperature)) +
  geom_point()

"Las variables de presión a nivel del agua también tienen correlación con las
temperaturas. La presión atmosférica disminuye con el aumento de la elevación,
por lo que podría intuirse que la temperatura media disminuye en lugares con
poca altitud en este conjunto de datos."

ggplot(data, aes(x = Mean_temperature, y = Sea_level_pressure)) +
  geom_point()

"Vamos a crear dos nuevas variables que resuman la información de max_temperature,
min_temperature y las variables relativas a presión."

data <- data %>%
  mutate(
    Diff_temperature = Max_temperature + Min_temperature,
    Diff_pressure = Standard_pressure - Sea_level_pressure
  ) %>%
  select(-Min_temperature, -Max_temperature, -Sea_level_pressure, -Standard_pressure)

"Vamos a realizar un análisis de comprobación de outliers multivariantes, es
decir, en combinación con múltiples variables. Primero, comprobamos
la normalidad con un test multivariante."

mvn(data = data, mvnTest = "hz")

"No lo pasa, por lo que los datos no siguen una normal con un alto grado de
confianza en altas dimensiones. Vamos a usar Mahalanobis igualmente para ver
la proporción de potenciales outliers"

mahal_dist <- mahalanobis(data, colMeans(data), cov(data))
threshold <- qchisq(0.975, df = dim(data)[2])  # 97.5% confidence level

data <- as.data.frame(data) %>% mutate(Outlier = mahal_dist > threshold)

# % of potential outliers
(nrow(data %>% filter(Outlier)) / nrow(data) * 100)

ggplot(as.data.frame(mahal_dist), aes(x = mahal_dist)) +
  geom_histogram(
    binwidth = 0.5,
    fill = "lightblue",
    color = "black",
    alpha = 0.7
  ) +
  labs(title = "Mahalanobis distances distribution", x = "Mahalanobis Dist", y = "Freq") +
  theme_minimal()

"Tal y como se observa en el gráfico, la distribución de las distancias tampoco
sigue una normal. Es un indicativo más de que los datos no son normales. Por ello
no voy a eliminar las variables encontradas por este método."

"Por último vamos a usar PCA para obtener un resumen de los datos y gráficas que
poder analizar más que por reducir dimensionalidad, ya que solo tenemos 10
variables."

pca_res <- prcomp(data)
pca_df <- data.frame(pca_res$x)
ggplot(pca_df, aes(x = PC1, y = PC2)) +
  geom_point(color = "blue") +
  labs(title = "PCA - First Two Principal Components", x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Get the variance of each component and visualize it
explained_variance <- summary(pca_res)$importance[2, ]
cumulative_variance <- cumsum(explained_variance)

var_explained_df <- data.frame(
  Component = 1:length(explained_variance),
  Variance_Explained = explained_variance,
  Cumulative_Variance = cumulative_variance
)

# We visualize the variance explained by each component
ggplot(var_explained_df, aes(x = Component)) +
  geom_bar(
    aes(y = Variance_Explained),
    stat = "identity",
    fill = "blue",
    alpha = 0.7
  ) +
  geom_line(aes(y = Cumulative_Variance * max(Variance_Explained)),
            color = "red",
            size = 1) +
  labs(title = "Variance Explained by PCA Components", x = "Principal Component", y = "Variance Explained") +
  scale_y_continuous(sec.axis = sec_axis(~ . / max(var_explained_df$Variance_Explained), name = "Cumulative Variance")) +
  theme_minimal()

"Dada esta gráfica podemos observar que la varianza acumulada según se añaden
más componentes deja de aumentar significativamente a partir de cuatro más o menos.
De todas formas, con solo dos (para poder visualizarlo) ya se obtienen una varianza
alta y suficiente para obtener cierta información."

# Biplot
autoplot(
  pca_res,
  data = data,
  colour = 'Mean_temperature',
  loadings = TRUE,
  loadings.label = TRUE,
  loadings.label.size = 3
)

"En este biplot se puede observar como temperaturas muy bajas se dan en lugares
con presiones altas y vientos altos, ya que los valores que se encuentran en
esa zona son combinaciones de valores altos para esas variables. Obviamente
valors bajos de las variables de temperatura (max, min) contribuyen muchísimo
a la temperatura media, pero por ser ta obvios son menos interesantes."

