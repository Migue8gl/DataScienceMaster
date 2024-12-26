library(caret)
library(kknn)
library("MASS")
library(tidyverse)

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
summary(data)

"Normalizamos los datos usando el escalado z-score. Esto hace que los datos sigan
una distribución con media 0 y desviación típica 1. Esto es muy necesario cuando
se tienen datos en distintas escalas, como es el caso, y se utilizan algoritmos
que usan distancias."

data <- data %>% mutate(Class = as.factor(Class))
class_column = data$Class
data <- as.data.frame(scale(data[1:ncol(data) - 1]))
data$Class <- class_column

data <- data %>%
  mutate(
    Thyroidstimulating = log1p(Thyroidstimulating),
    TSH_value = log1p(TSH_value)
  )
summary(data)

"Vamos a dividir el conjunto d datos en conjuntos de train y test. Para eso definimos
la siguiente función"

train_test_split <- function(data,
                             test_percentage = 0.2,
                             seed = 42) {
  set.seed(seed)
  data <- data[sample(1:nrow(data)), ]
  
  n <- round((1 - test_percentage) * nrow(data))
  
  train <- data[1:n, 1:ncol(data)]
  test <- data[(n + 1):nrow(data), 1:ncol(data)]
  
  list(train = train, test = test)
}

results <- train_test_split(data)
train <- results$train
test <- results$test
str(results)

"Una vez divididos los conjuntos podemos empezar a experimentar con KNN."
fit1 <- kknn(Class ~ .,
             train = train,
             test = test,
             k = 1)

make_confusion_matrix <- function(knn_model) {
  predicted <- fitted(knn_model)
  conf_matrix <- table(Predicted = predicted, Actual = test$Class)
  conf_matrix_df <- as.data.frame(as.table(conf_matrix))
  
  plot <- ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%d", Freq)),
              color = "white",
              size = 10) +
    scale_fill_gradient(low = "cyan", high = "darkgreen") +
    theme_minimal() +
    theme(
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      plot.title = element_text(
        size = 14,
        face = "bold",
        hjust = 0.5
      )
    ) +
    labs(
      title = "Confusion Matrix",
      x = "Actual Class",
      y = "Predicted Class",
      fill = "Count"
    ) +
    coord_equal()
  
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  print(plot)
  print(paste("Accuracy:", round(accuracy, 2)))
}

make_confusion_matrix(fit1)

"Dadas estas métricas parece que el knn está generalizando muy bien con k=1, lo
cual es bastante asombroso, pues k=1 es un valor muy dado al sobreajuste ya que
el modelo se hace mucho más sensible a cambios cuando añadimos nuevos datos, se
hace muy variable."

"Vamos a probar varios valores de k para ver como varían los resultados."
fit2 <- kknn(Class ~ .,
             train = train,
             test = test,
             k = 5)

make_confusion_matrix(fit2)


fit3 <- kknn(Class ~ .,
             train = train,
             test = test,
             k = 15)

make_confusion_matrix(fit3)


fit4 <- kknn(Class ~ .,
             train = train,
             test = test,
             k = 40)

make_confusion_matrix(fit4)

fit5 <- kknn(Class ~ .,
             train = train,
             test = test,
             k = 100)

make_confusion_matrix(fit5)

"El modelo con k cada vez mayor pierde precisión, porque al hacer esto, el
modelo se suaviza y se vuelve más general. Esto puede reducir el sobreajuste
al hacer que las predicciones dependan de un número mayor de vecinos, lo que
ayuda a promediar los datos y a reducir la sensibilidad a las fluctuaciones
pequeñas. Sin embargo, si kk es demasiado grande, el modelo comienza a perder
capacidad para captar patrones específicos, lo que puede llevar a un subajuste
(underfitting), ya que los puntos de datos más cercanos pueden no tener tanta
influencia sobre la predicción final. "

"En mi caso escogería k=5. Aunque k=1 sea mejor y me haya dado mejores resultados
en test, añadir un poco de regularización en este caso, considero que no devalúa
demasiado la calidad del modelo y previene de posibles ajustes demasiado finos. Si
bien es verdad que el resultado de test es mejor para k=1 y por tanto el modelo,
incluso con k=1 parece no sobreajustar nada en datos nunca vistos, prefiero ser
precavido."

"Ahora vamos a usar LDA para ajustar un modelo al conjunto de datos. Pero primero
han de comprobarse ciertas asunciones"

"Los datos son normales para cada clase."
str(data)
test_variable <- function(var_name) {
  resultados <- data %>%
    group_by(Class) %>%
    summarize(
      p_value = if (is.numeric(.data[[var_name]]) &&
                    all(!is.na(.data[[var_name]]))) {
        shapiro.test(.data[[var_name]])$p.value
      } else {
        NA
      },
      normal = ifelse(p_value > 0.05, "Puede ser normal", "No es normal"),
      .groups = "drop"
    )
  plot <- ggplot(data, aes(sample = .data[[var_name]])) +
    stat_qq() + stat_qq_line() +
    facet_wrap(~ Class)
  print(plot)
  print(resultados)
}

test_variable("T3resin")
test_variable("Thyroxin")
test_variable("Triiodothyronine")
test_variable("Thyroidstimulating")
test_variable("TSH_value")

"Parece que el test de Shapiro no rechaza para algunas variables dentro de algunas
clases. Si bien no se puede decir que no son normales, tampoco se pueden rechazar.
Dada esta evidencia y la de los qqplots, no se verifica la primera asunción de LDA."

"Para comprobar si cada clase tiene matrices de varianze-covarianza idénticas, se
puede utilizar el test de Bartlett"

bartlett.test(T3resin ~ Class, data)
bartlett.test(Thyroxin ~ Class, data)
bartlett.test(Triiodothyronine ~ Class, data)
bartlett.test(Thyroidstimulating ~ Class, data)
bartlett.test(TSH_value ~ Class, data)

"Se rechaza el test de Bartlett para todas las variables, lo que significa que
las varianzas de los grupos o muestras comparadas no son iguales, es decir, no
se cumple el supuesto de homogeneidad de varianzas (homocedasticidad)."

"Clasificamos con LDA"
lda_model <- lda(Class ~ ., data = train)
lda_model
lda.pred.train <- predict(lda_model, train)
lda.pred.test <- predict(lda_model, test)

plot_data <- lda.pred.train$x %>%
  as_tibble() %>%
  mutate(Class = train$Class)

"Mostramos el gráfico donde se muestra como se distribuyen las observaciones de
las clases en el espacio generado por LDA utilizando las primeras dos componentes
lineales discriminantes (LD1 y LD2)."

"Se puede observar que las clases están bien separadas en el espacio generado por
LDA. Una buena separación indica que el modelo ha logrado discriminar
correctamente entre clases."

ggplot(data = plot_data) +
  geom_point(aes(x = LD1, y = LD2, color = Class)) +
  scale_colour_manual(
    name = "Class",
    values = c("red", "green", "blue"),
    labels = c("1", "2", "3")
  ) +
  labs(title = "Data Transformed After LDA")

t <- table(lda.pred.test$class, test$Class)
t

sum(diag(t)) / nrow(test)

plot_data <- lda.pred.test$x %>%
  as_tibble() %>%
  mutate(known = test$Class, # Replace with the correct column for class labels
         prediction = lda.pred.test$class) %>%
  pivot_longer(c("prediction", "known"),
               names_to = "Type",
               values_to = "Class")

ggplot(data = plot_data) +
  geom_point(aes(
    x = LD1,
    y = LD2,
    shape = Type,
    color = Class
  )) +
  scale_colour_manual(
    name = "Species",
    values = c("red", "green", "blue"),
    labels = c("1", "2", "3")
  ) +
  scale_shape_manual(name = "Type", values = c(5, 3)) +
  labs(title = "Validation Data + Predictions Transformed After LDA")

"Pese a no cumplirse las normalidades de datos por clase y la igualdad de matrices
de covarianza-varianza, el modelo es capaz de generalizar muy bien y de encontrar
separabilidad entre las clases."

qda_model <- qda(Class ~ ., data = train)
qda_model
qda.pred.train <- predict(qda_model, train)
qda.pred.test <- predict(qda_model, test)

t <- table(Predicted = qda.pred.test$class, Actual = test$Class)
t

sum(diag(t)) / nrow(test)

conf_matrix_df <- as.data.frame(as.table(t))

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), color = "white", size = 10) +
  scale_fill_gradient(low = "cyan", high = "darkgreen") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  ) +
  labs(title = "Confusion Matrix",
       x = "Actual Class",
       y = "Predicted Class",
       fill = "Count") +
  coord_equal()

"Vamos a comparar los tres algoritmos."

k_fold_cross_validation <- function(data,
                                    k = 10,
                                    model_function,
                                    metric_function,
                                    seed = 42) {
  set.seed(seed)
  folds <- sample(1:k, size = nrow(data), replace = TRUE)
  performance_metrics <- c()
  
  for (i in 1:k) {
    test_indices <- which(folds == i)
    train_indices <- setdiff(1:nrow(data), test_indices)
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    model <- model_function(train_data, test_data)
    if (inherits(model, "kknn")) {
      predictions <- fitted(model)
    } else {
      predictions <- predict(model, test_data)
    }
    
    if (inherits(model, "lda") | inherits(model, "qda")) {
      predictions_class <- predictions$class
    } else {
      predictions_class <- predictions
    }
    
    performance <- metric_function(predictions_class, test_data$Class)
    performance_metrics <- c(performance_metrics, performance)
  }
  
  mean_performance <- mean(performance_metrics)
  std_deviation <- sd(performance_metrics)
  
  return(list(mean = mean_performance, std_deviation = std_deviation))
}

model_function_knn <- function(train_data, test_data) {
  model <- kknn(Class ~ .,
                train = train_data,
                test = test_data,
                k = 5)
  return(model)
}

model_function_lda <- function(train_data, test_data) {
  model <- lda(Class ~ ., data = train_data)
  return(model)
}

model_function_qda <- function(train_data, test_data) {
  model <- qda(Class ~ ., data = train_data)
  return(model)
}

metric_function_accuracy <- function(predictions, actual) {
  confusion_matrix <- table(predicted = predictions, actual = actual)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(accuracy)
}

result_knn <- k_fold_cross_validation(
  data,
  k = 10,
  model_function = model_function_knn,
  metric_function = metric_function_accuracy
)
print(paste(
  "KNN Accuracy: ",
  round(result_knn$mean, 4),
  "±",
  round(result_knn$std_deviation, 4)
))

result_lda <- k_fold_cross_validation(
  data,
  k = 10,
  model_function = model_function_lda,
  metric_function = metric_function_accuracy
)
print(paste(
  "LDA Accuracy: ",
  round(result_lda$mean, 4),
  "±",
  round(result_lda$std_deviation, 4)
))

result_qda <- k_fold_cross_validation(
  data,
  k = 10,
  model_function = model_function_qda,
  metric_function = metric_function_accuracy
)
print(paste(
  "QDA Accuracy: ",
  round(result_qda$mean, 4),
  "±",
  round(result_qda$std_deviation, 4)
))
