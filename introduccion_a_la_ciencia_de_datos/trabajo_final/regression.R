library(tidyverse)
library(kknn)
library(RWeka)

"Leemos los datos saltandonos las cabeceras iniciales y después las parseamos
a mano."

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
str(data)

"Vamos a aplicar las transformaciones realizadas en la parte del EDA."

data <- as.data.frame(scale(data))
data$Diff_temperature <- data$Max_temperature + data$Min_temperature
data$Diff_pressure <- data$Standard_pressure - data$Sea_level_pressure

data <- data[, !names(data) %in% c("Min_temperature",
                                   "Max_temperature",
                                   "Sea_level_pressure",
                                   "Standard_pressure")]

summary(data)
pairs(data)

"Ahora vamos a utilizar las 5 variables regresoras que más sentido tengan según
nuestro criterio. Según lo visto en el EDA, en lo relativo a correlación y al
propio sentido común, la variable más prometedora es Diff_temperature (es la
combinación lineal de Max_temperature y Min_temperature). Seguido de Dewpoint,
Diff_pressure (combinación lineal de variables originales de presión), Visibility
y Wind_speed.

Se ha tenido en cuenta las variables que parecen más lineales con respecto a la
variable objetivo, o que al menos parezcan tener una relación en algún sentido (por
ejemplo, Visibility es algo curva, algo logaritmica)."

perform_linear_regression <- function(data, target_var = "Mean_temperature", predictors) {
  regression_results <- list()
  
  for (predictor in predictors) {
    formula <- as.formula(paste(target_var, "~", predictor))
    
    fit <- lm(formula, data = data)
    
    plot_data <- data.frame(x = data[[predictor]],
                            y = data[[target_var]],
                            fitted = predict(fit))
    
    p <- ggplot(plot_data, aes(x = x, y = y)) +
      geom_point(alpha = 0.5) +
      geom_line(aes(y = fitted), color = "blue", size = 1) +
      labs(
        title = paste(target_var, "vs", predictor),
        x = predictor,
        y = target_var
      ) +
      theme_minimal()
    
    regression_results[[predictor]] <- list(model = fit,
                                            summary = summary(fit),
                                            plot = p)
    
    # Print summary and plot
    cat("\n--- Regression Results for", predictor, "---\n")
    print(regression_results[[predictor]]$summary)
    print(regression_results[[predictor]]$plot)
  }
  
  return(regression_results)
}

predictors <- c("Diff_temperature",
                "Dewpoint",
                "Diff_pressure",
                "Visibility",
                "Wind_speed")
regression_analysis <- perform_linear_regression(data, predictors = predictors)

"Claramente el mejor predictor de la temperatura media es diff_temperature, seguido
de Dewpoint. El RSE es extremadamente bajo usando ese predictor, además tiene un
R cuadrado ajustado del 0.98, lo que significa que con solo esa variable es posible
explicar el 98% de la variabilidad de la variable objetivo."

"Vamos a eliminar variables usando backwards selection."

fit1 <- data %>%
  lm(formula = Mean_temperature ~ ., data = .)

summary(fit1)

fit2 <- data %>%
  lm(formula = Mean_temperature ~ . - Max_wind_speed, data = .)

summary(fit2)

fit3 <- data %>%
  lm(formula = Mean_temperature ~ . - Max_wind_speed - Precipitation,
     data = .)

summary(fit3)

"A continuación vamos a ajustar un modelo de regresión lineal múltiple utilizando
varias variables y teniendo en cuenta relaciones no linales e interacciones. Aunque
los resultados son difícilmente mejorables"

fit1 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature, data = .)

summary(fit1)


fit2 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature + Dewpoint + Diff_pressure,
     data = .)

summary(fit2)

fit3 <- data %>%
  lm(
    formula = Mean_temperature ~ Diff_temperature + Dewpoint + Diff_pressure + Visibility,
    data = .
  )

summary(fit3)

"Añadir poco a poco todas las variables mejora el resultado, pero es mínimo. No creo
que sea un camino a seguir. Voy a probar con interacciones entre variables que
tengan sentido."

fit4 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature * Dewpoint,
     data = .)

summary(fit4)

fit5 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature * I(Dewpoint ^ 2),
     data = .)

summary(fit5)

fit6 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature * I(Dewpoint ^ 2) * Dewpoint,
     data = .)

summary(fit6)

fit7 <- data %>%
  lm(
    formula = Mean_temperature ~ Diff_temperature * I(Dewpoint ^ 2) * Dewpoint * I(Diff_temperature ^
                                                                                     2),
    data = .
  )

summary(fit7)

"Parece que la interacción de las dos variables más importantes junto a no linealidad
mejora más el resultado."

"Voy a probar con Visibilidad al cuadrado, ya que pareciese por la gráfica que
sigue una relación no lineal con respecto a la variable objetivo."

fit8 <- data %>%
  lm(formula = Mean_temperature ~ Diff_temperature + I(Visibility ^ 2),
     data = .)

summary(fit8)

"Probando varias combinaciones, lo que he podido ver es que el modelo parece
ajustar mejor añadiendo interacciones entre varias variables en vez de solo
añadirlas. Alguna no linealidad (sobre todo con Dewpoint) parece añadir algo
de mejora. Voy a usar el modelo final del método backwards por la interpretabilidad
y porque el mejor modelo que usa interacciones y no linealidad (fit7) no mejora tanto
como para plantearse sacrificar interpretabilidad."

"Ahora vamos a aplicar regresión con el algoritmo kNN."

fitknn1 <- kknn(Mean_temperature ~ ., data, data)

"Visualizamos los datos originales vs los puntos ajustados del knn"
ggplot(data = data, aes(x = Diff_temperature, y = Mean_temperature)) +
  geom_point() +
  geom_point(aes(y = fitknn1$fitted.values),
             color = "blue",
             shape = 20) +
  labs(x = "Diff_temperature", y = "Mean_temperature", title = "kNN") +
  theme_minimal()

"Calculamos el RMSE"
yprime <- fitknn1$fitted.values
sqrt(sum((data$Mean_temperature - yprime) ^ 2) / length(yprime))

"Utilizando todos los datos podemos obtener un RMSE prácticamente igual que en
regresión usando solo la variable de diff_temperature. Este problema concreto
parece bastante apto para una regresión lineal, de todas formas probaré
algunas combinaciones más."

fitknn2 <- kknn(Mean_temperature ~ Diff_temperature, data, data)

"Calculamos el RMSE"
yprime <- fitknn2$fitted.values
sqrt(sum((data$Mean_temperature - yprime) ^ 2) / length(yprime))

"Usando solo la variable más importante es capaz de conseguir un mejor ajuste
que la regresión lineal."

ggplot(data = data, aes(x = Diff_temperature, y = Mean_temperature)) +
  geom_point() +
  geom_point(aes(y = fitknn2$fitted.values),
             color = "blue",
             shape = 20) +
  labs(x = "Diff_temperature", y = "Mean_temperature", title = "kNN") +
  theme_minimal()

"Podemos apreciar mucho menos ruido que usando todas las variables."

fitknn3 <- kknn(Mean_temperature ~ Diff_temperature + Dewpoint, data, data)

"Calculamos el RMSE"
yprime <- fitknn3$fitted.values
sqrt(sum((data$Mean_temperature - yprime) ^ 2) / length(yprime))

ggplot(data = data, aes(x = Diff_temperature, y = Mean_temperature)) +
  geom_point() +
  geom_point(aes(y = fitknn3$fitted.values),
             color = "blue",
             shape = 20) +
  labs(x = "Diff_temperature", y = "Mean_temperature", title = "kNN") +
  theme_minimal()

"Añadiendo Dewpoint se consigue un ajuste con menor ruido incluso. Con diff_temperature
se podía modelar una línea muy buena que ajustase los datos, pero utilizando además
dewpoint parece que la varianza se modela mejor."

fitknn4 <- kknn(Mean_temperature ~ Diff_temperature * Dewpoint * I(Dewpoint ^
                                                                     2),
                data,
                data)

"Calculamos el RMSE"
yprime <- fitknn4$fitted.values
sqrt(sum((data$Mean_temperature - yprime) ^ 2) / length(yprime))

ggplot(data = data, aes(x = Diff_temperature, y = Mean_temperature)) +
  geom_point() +
  geom_point(aes(y = fitknn4$fitted.values),
             color = "blue",
             shape = 20) +
  labs(x = "Diff_temperature", y = "Mean_temperature", title = "kNN") +
  theme_minimal()

"Si en vez de sumar las variables añadimos interacciones entre ellas y además tenemos
en cuenta un poco de no linealidad mejora el ajuste."

"He probado con otras combinaciones, pero no parece que encuentre nada mejor. Además
hay que tener en cuenta que quizá estemos ajustando demasiado los resultados a la
muestra, lo cuál en un entrenamiento real llevaría al sobreajuste."

"Vamos a realizar 5-fold cross validation con el mejor modelo obtenido de regresión
lineal múltiple y a partir de ahí, intentar obtener un mejor modelo con kNN. Después
compararemos todos los resultamos entre sí y con el algoritmo M5."

run_k_fold_cv <- function(data, model = "lm", k = 5) {
  columns <- c(
    "Dewpoint",
    "Precipitation",
    "Max_temperature",
    "Min_temperature",
    "Sea_level_pressure",
    "Standard_pressure",
    "Visibility",
    "Wind_speed",
    "Max_wind_speed",
    "Mean_temperature"
  )
  
  colnames(data) <- columns
  
  train_mse_list <- numeric()
  test_mse_list <- numeric()
  
  data$Diff_temperature <- data$Max_temperature + data$Min_temperature
  data$Diff_pressure <- data$Standard_pressure - data$Sea_level_pressure
  
  data <- data[, !names(data) %in% c("Min_temperature",
                                     "Max_temperature",
                                     "Sea_level_pressure",
                                     "Standard_pressure")]
  idx <- sample(nrow(data))
  data <- data[idx, ]
  
  fold_size <- round(nrow(data) / k)
  
  results <- data.frame(
    Actual = numeric(),
    Predicted = numeric(),
    Fold = numeric(),
    MSE = numeric(),
    Train = numeric(),
    Test = numeric()
  )
  
  for (i in 0:(k - 1)) {
    start <- 1 + i * fold_size
    end <- min((i + 1) * fold_size, nrow(data))
    
    val <- data[start:end, ]
    train <- data[-(start:end), ]
    
    train_scaled <- scale(train)
    mean_train <- attr(train_scaled, "scaled:center")
    sd_train <- attr(train_scaled, "scaled:scale")
    
    train <- as.data.frame(train_scaled)
    
    val <- as.data.frame(scale(val, center = mean_train, scale = sd_train))
    
    if (model == "lm") {
      fit <- lm(Mean_temperature ~ ., data = train)
      train_pred <- predict(fit, newdata = train)
      yprime <- predict(fit, newdata = val)
    } else if (model == "kknn") {
      fit <- kknn(Mean_temperature ~ ., train = train, test = val)
      fit_train <- kknn(Mean_temperature ~ ., train = train, test = train)
      train_pred <- fit_train$fitted.values
      yprime <- fit$fitted.values
    } else if (model == "m5") {
      fit <- M5P(Mean_temperature ~ ., data = train)
      train_pred <- predict(fit, newdata = train)
      yprime <- predict(fit, newdata = val)
    }
    
    train_mse <- mean((train$Mean_temperature - train_pred) ^ 2)
    test_mse <- mean((val$Mean_temperature - yprime) ^ 2)
    
    train_mse_list <- c(train_mse_list, train_mse)
    test_mse_list <- c(test_mse_list, test_mse)
    
    results <- rbind(
      results,
      data.frame(
        Actual = val$Mean_temperature,
        Predicted = as.numeric(yprime),
        Fold = i + 1,
        MSE = test_mse,
        Train = tail(train_mse_list, 1),
        Test = tail(test_mse_list, 1)
      )
    )
  }
  
  results
}

data <- read.csv("wankara/wankara.dat", skip = 14, header = FALSE)
lm_results <- run_k_fold_cv(data, model = "lm")
knn_results <- run_k_fold_cv(data, model = "kknn")
m5_results <- run_k_fold_cv(data, model = "m5")

# Summary of results
lm_mse <- mean(lm_results$MSE)
knn_mse <- mean(knn_results$MSE)
m5_mse <- mean(m5_results$MSE)

lm_mse_train <- mean(lm_results$Train)
knn_mse_train <- mean(knn_results$Train)
m5_mse_train <- mean(m5_results$Train)

print(paste("Linear Regression MSE:", lm_mse))
print(paste("KNN MSE:", knn_mse))
print(paste("M5 MSE:", m5_mse))

plot_predictions <- function(results, model) {
  ggplot(results, aes(
    x = Actual,
    y = Predicted,
    color = as.factor(Fold)
  )) +
    geom_point(alpha = 0.7) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "black"
    ) +
    labs(
      title = paste(model, "-", "Results"),
      x = "Actual Mean Temperature",
      y = "Predicted Mean Temperature",
      color = "Fold"
    ) +
    theme_minimal()
}

plot_lm <- plot_predictions(lm_results, "Linear Regression")
plot_knn <- plot_predictions(knn_results, "KNN")
plot_m5 <- plot_predictions(m5_results, "M5")

print(plot_lm)
print(plot_knn)
print(plot_m5)

"Por los resultados de test y lo visto en las gráficas, parece que el modelo que
mejor se ajusta a este dataset es regresión lineal múltiple seguido muy de cerca por knn."

"Vamos a comparar los algoritmos."

resultados_test <-  read.csv("regr_test_alumnos.csv")
resultados_train <- read.csv("regr_train_alumnos.csv")
tablatst <- cbind(resultados_test[, 2:dim(resultados_test)[2]])
colnames(tablatst) <- names(resultados_test)[2:dim(resultados_test)[2]]
rownames(tablatst) <- resultados_test[, 1]
tablatra <- cbind(resultados_train[, 2:dim(resultados_train)[2]])
colnames(tablatra) <- names(resultados_train)[2:dim(resultados_train)[2]]
rownames(tablatra) <- resultados_train[, 1]

tablatst[17, 1] <- lm_mse
tablatst[17, 2] <- knn_mse
tablatst[17, 3] <- m5_mse

tablatra[17, 1] <- lm_mse_train
tablatra[17, 2] <- knn_mse_train
tablatra[17, 3] <- m5_mse_train

"Normalizamos utilizando las diferencias relativas entre los resultados de los
algoritmos. Después, generamos una tabla con valores ajustados y procedemos al
test de Wilcoxon, que es un test por pares."

difs <- (tablatst[, 1] - tablatst[, 2]) / tablatst[, 1]
wilc_1_2 <- cbind(ifelse (difs < 0, abs(difs) + 0.1, 0 + 0.1),
                  ifelse (difs > 0, abs(difs) + 0.1, 0 + 0.1))
colnames(wilc_1_2) <- c(colnames(tablatst)[1], colnames(tablatst)[2])
head(wilc_1_2)

LMvsKNNtst <- wilcox.test(wilc_1_2[, 1],
                          wilc_1_2[, 2],
                          alternative = "two.sided",
                          paired = TRUE)
Rmas <- LMvsKNNtst$statistic
pvalue <- LMvsKNNtst$p.value
LMvsKNNtst <- wilcox.test(wilc_1_2[, 2],
                          wilc_1_2[, 1],
                          alternative = "two.sided",
                          paired = TRUE)
Rmenos <- LMvsKNNtst$statistic
Rmenos
Rmas
pvalue

"Dado un p-valor de menos de 0.7, no se puede recharzar la hipótesis nula, por lo que
no podemos asegurar que existen diferencias estadísticamente significativas entre
KNN y LM."

"Ahora realizamos test múltiples para comparar todos los algoritmos entre sí. Para
ello usamos el test de friedman."

test_friedman <- friedman.test(as.matrix(tablatst))
test_friedman

"Se aplica una corrección para evitar errores acumulados."

tam <- dim(tablatst)
groups <- rep(1:tam[2], each = tam[1])
pairwise.wilcox.test(as.matrix(tablatst),
                     groups,
                     p.adjust = "holm",
                     paired = TRUE)

"Esto indica que el algoritmo 3 (M5) tiene evidencia estadística de ser mejor sobre
los otros dos algoritmos. En este test además se puede ver como el algoritmo 1 y el
2 (KNN vs LM) tienen también diferencias significativas entre ellos, como se pudo
ver en el anterior análisis."

"Vamos a hacer lo mismo con los resultados de train."

difs <- (tablatra[, 1] - tablatra[, 2]) / tablatra[, 1]
wilc_1_2 <- cbind(ifelse (difs < 0, abs(difs) + 0.1, 0 + 0.1),
                  ifelse (difs > 0, abs(difs) + 0.1, 0 + 0.1))
colnames(wilc_1_2) <- c(colnames(tablatra)[1], colnames(tablatra)[2])
head(wilc_1_2)

LMvsKNNtst <- wilcox.test(wilc_1_2[, 1],
                          wilc_1_2[, 2],
                          alternative = "two.sided",
                          paired = TRUE)
Rmas <- LMvsKNNtst$statistic
pvalue <- LMvsKNNtst$p.value
LMvsKNNtst <- wilcox.test(wilc_1_2[, 2],
                          wilc_1_2[, 1],
                          alternative = "two.sided",
                          paired = TRUE)
Rmenos <- LMvsKNNtst$statistic
Rmenos
Rmas
pvalue

sapply(tablatra, median)
sapply(tablatst, median)

test_friedman <- friedman.test(as.matrix(tablatra))
test_friedman

tam <- dim(tablatra)
groups <- rep(1:tam[2], each = tam[1])
pairwise.wilcox.test(as.matrix(tablatra),
                     groups,
                     p.adjust = "holm",
                     paired = TRUE)

"Lo esperado en este apartado era que se obtuvieran resultados más significativos.
Si los algoritmos iban bien en test, en training deberían haber ido incluso mejor.
Aunque esta premisa no es cierta siempre, en este caso puede observarse como
los p-valores son más bajos incluso. Por ello, parece que los mismos algoritmos siguen
siendo los mejores y no parece que se haya realizan un sobreajuste."

