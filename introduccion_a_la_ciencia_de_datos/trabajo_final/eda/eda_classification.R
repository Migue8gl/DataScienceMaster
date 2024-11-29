library(tidyverse)
library(moments)
library("corrplot")

"Leemos los datos saltandonos las cabeceras iniciales y después las parseamos
a mano. Seguido, vamos a ver unos pocos datos con head para hacernos una primera
idea."

data <- read.csv("newthyroid/newthyroid.dat",
                 skip = 10,
                 header = FALSE)
colnames(data) <- c(
  "T3resin",
  "Thyroxin",
  "Triiodothyronine",
  "Thyroidstimulating",
  "TSH_value",
  "Class"
)
head(data)

"Con la función de summary obtenemos un resumen estadístico general de cada
columna. Además vamos a obtener otros datos informativos como la dimensión,
y la estructura."

summary(data)
dim(data)
str(data)
colSums(is.na(data))

"Por lo que podemos ver, existen 6 variables o columnas (o características) para
este conjunto. Existe gran variabilidad en características como TSH_value y
Thyroidstimulating, donde valores muy bajos son cercanos al cero y muy altos
superan los 50.

Además, la mayoría de valores en estas variables parecen ser que rondan valores
bajos, esto nos lo puede indicar el tercer cuartil, que es en ambas variables
muy bajo en comparación al valor máximo que toman.

Las variables son todas númericas flotantes, excepto T3resin que es entera. No
hay valores faltantes en ninguna columna."

"Las clases a predecir en este problema de clasificación son las siguientes:"

table(data$Class)
prop.table(table(data$Class))

"Se puede observar un desbalanceo en el número de ejemplos para cada clase. La
clase 1 es muy mayoritaria."

"Vamos a normalizar todas las variables. Las que más necesitan esta
transformación son las anteriormente mencionadas, que tienen valores muy
separados en el espacio. Las clases además, deben transformarse en factores
para poder tratarlas como categorías."

data <- data %>%
  mutate(Class = as.factor(Class)) %>%
  mutate(across(where(is.numeric), ~ (. - mean(., na.rm = TRUE)) / sd(.)))

selected_columns <- c("T3resin",
                      "Thyroxin",
                      "Triiodothyronine",
                      "Thyroidstimulating",
                      "TSH_value")

long_data <- data %>%
  pivot_longer(cols = selected_columns,
               names_to = "Variable",
               values_to = "Value")

ggplot(long_data, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  labs(title = "Boxplots", x = "Variable", y = "Value") +
  theme_minimal()

"Dada esta gráfica de boxplots podemos observar aquellos valores que se alejan
demasiado de la distribución de los datos. De hecho, el método de distancia
intercuartil es muy útil para eliminar outliers."

q3 <- quantile(data$TSH_value, 0.75)
iqr <- IQR(data$TSH_value)
upper_bound <- q3 + 1.5 * iqr

# Porcentaje de "outliers" en TSH_value
length(data$TSH_value[data$TSH_value > upper_bound]) / length(data$TSH_value)

"Usando el método de distancia intercuartil (solo por arriba, ya que en el
boxplot podemos ver que no hay ninguno por abajo) encontramos potenciales
outliers. El porcentaje de ellos para la característica seleccionada es muy
pequeño. En el caso de que estuviese seguro (conocimiento de dominio) de que
esos valores son irreales, los quitaría, pero realmente no lo sé y el
quitarlos ahora me evitaría un posible futuro análisis con el modelo de
predicción usado, ya que a priori no sé si van a afectar severamente el
rendimiento de mi modelo."

"Creamos una pequeña función para poder mostrar un gráfico de barras, de esta
forma podremos ver fácilmente las posibles distribuciones de cada variable."

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

plot_distribution(data, "TSH_value")

"Claramente tiene una asimetría a la izquierda."

plot_distribution(data, "T3resin")

"T3resin tiene un aspecto que parece bastante normal. Ligeramente asimétrica
hacia la derecha."

plot_distribution(data, "Thyroxin")

"Thyroxin también tiene un aspecto que parece bastante normal."

plot_distribution(data, "Triiodothyronine")

"Triiodothyronine tiene una asimetría a la izquierda."

plot_distribution(data, "Thyroidstimulating")

"Thyroidstimulating tiene una asimetría a la izquierda muy acentuada."

"Dadas estas gráficas de densidades de cada variable, podemos observar como se
distribuyen sus valores. Vemos que hay unas cuantas que cuentan con asimetría
extrema. Estos que presentan esta asimetría son aquellas variables que
anteriormente habíamos identificado con pocos valores extremos y muchos
valores bajos."

ggplot(long_data, aes(sample = Value)) +
  stat_qq() +
  stat_qq_line() +
  facet_wrap( ~ Variable) +
  labs(title = "Qqplot of every variable", y = "Variable quantiles", x = "Theoretical quantiles") +
  theme_minimal()

"T3resin y Thyroxin son las que se parecen más a una normal, pero aún así tienen
variaciones en las colas. El resto de variables son muy asimétricas, ya se había
diagnosticado, pero es una confirmación más.

Vamos a hacer un test de Shapiro sobre las dos variables más prometedoras para
ver si podemos rechazar la hipótesis nula de que no siguen una distribución
normal."

shapiro.test(data$T3resin)
shapiro.test(data$Thyroxin)

"El test de shapiro es muy significativo para las dos variables más
prometedoras, lo que indica que podemos rechazar las asumpciones de normalidad
que hemos estado construyendo por medio de las gráficas. Ninguna variable de las
que tenemos sigue una distribución normal, esto lo tendremos en cuenta para la
fase de clasificación."

"Vamos a hacer test de aimetría y de curtosis para las variables que sabemos
no son normales."

agostino.test(data$T3resin)
agostino.test(data$Thyroxin)
anscombe.test(data$T3resin)
anscombe.test(data$Thyroxin)

"Entendemos pues que los datos tienen asimetría y curtosis, pues se rechazan en
ambos casos."


"Creamos otra función para ver la distribución, pero por clase. De esta forma
podemos observar que distribuciones de valores parecen asociarse a ciertas
clases."

plot_distribution_for_every_class <- function(data, var_name, binwidth = 0.6) {
  ggplot(data, aes(x = !!sym(var_name), fill = Class)) +
    geom_density(color = "black", alpha = 0.7) +
    labs(
      title = paste("Distribution of", var_name),
      x = var_name,
      y = "Frequency"
    ) +
    facet_wrap(~ Class) +
    theme_minimal() +
    theme(legend.position = "none")
}

plot_distribution_for_every_class(data, "Thyroidstimulating")
plot_distribution_for_every_class(data, "TSH_value")
plot_distribution_for_every_class(data, "T3resin")
plot_distribution_for_every_class(data, "Thyroxin")
plot_distribution_for_every_class(data, "Triiodothyronine")

"Parece observarse que para todas las variables, valores altos de la misma
suelen agruparse en la clase 1."

"Vamos a mirar si existe alguna correlación entre variables, para ello podemos
usar dos métodos. Con pairs mostramos cada variable en relación a otra variable.
Lo bueno es que podemos de manera intuitiva identificar relaciones que vayan
más allá de la linealidad. Con la librería de correlación, podemos mostrar
gráficamente junto al coeficiente las correlaciones entre variables."

numeric_data <- select(data, -Class)
pairs(numeric_data)

"Una de las relaciones más notables parece estar entre TSH_value y
Thyroidstimulating, donde se observa una distribución bastante concentrada cerca
de los valores más bajos, con algunos valores atípicos.
Hay patrones interesantes entre T3resin y Thyroxin, que muestran una dispersión
no aleatoria, sugiriendo algún tipo de relación entre estas hormonas tiroideas.
Triiodothyronine también muestra patrones de agrupación interesantes con otras
variables, especialmente visible en algunas de las dispersiones."

cor_matrix <- cor(numeric_data)
corrplot(cor_matrix,
         method = "number",
         type = "upper",
         tl.col = "black")

"En el caso de este dataset, no encuentro ninguna correlación demasiado grande a
excepción de Triiodothyronine y Thyroxin, que presenten una correlación positiva
y algo lineal bastante interesante.
Dados estos resultados, yo no eliminaría ninguna por redundancia. Además, la
dimensión del problema no es tal como para tener que optimizar el tamaño
de las componentes / características que tenemos, pues hay muy pocas. De hecho
muchas veces, dos características muy correladas no implican redundancia. Un
ejemplo de esto aparece en Feature Extraction - Foundations and Applications
by I. Guyon et al. (p.10, figure 2 (e)), donde dos características con
un ratio de correlación alto son totalmente necesarias (ambas) para la
clasificación del problema."

"En cuanto a transformaciones, es posible que fuese muy útil en aquellas
aquellas variables con asimetrías largas a la derecha, en ese caso podríamos
aplicar transformaciones logarítmicas."

data <- data %>%
  mutate(
    log_Triiodothyronine = log1p(Triiodothyronine),
    log_Thyroidstimulating = log1p(Thyroidstimulating),
    log_TSH_value = log1p(TSH_value)
  )

plot_distribution(data, "log_Triiodothyronine")
plot_distribution(data, "Triiodothyronine")
plot_distribution(data, "log_Thyroidstimulating")
plot_distribution(data, "Thyroidstimulating")
plot_distribution(data, "log_TSH_value")
plot_distribution(data, "TSH_value")

shapiro.test(data$log_Triiodothyronine)
shapiro.test(data$Triiodothyronine)
agostino.test(data$log_Triiodothyronine)
agostino.test(data$Triiodothyronine)

shapiro.test(data$log_Thyroidstimulating)
shapiro.test(data$Thyroidstimulating)
agostino.test(data$log_Thyroidstimulating)
agostino.test(data$Thyroidstimulating)

shapiro.test(data$log_TSH_value)
shapiro.test(data$TSH_value)
agostino.test(data$log_TSH_value)
agostino.test(data$TSH_value)

"Se ha reducido la asimetría notablemente, aunque eso no hace que las
distribuciones de las variables sean normales. "