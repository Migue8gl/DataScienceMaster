library(arules)
library(tidyverse)

"El dataset ha sido obtenido de Kaggle."
"https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised"

"Descripción de las columnas:

Balance:
El saldo actual del cliente en la tarjeta de crédito.

Balance Frequency:
Frecuencia con la que el cliente mantiene un saldo en su tarjeta de crédito. Podría ser un valor entre 0 y 1, donde valores más altos indican un uso más consistente.

Purchases:
El monto total de compras realizadas por el cliente en los últimos 6 meses.

One-off Purchases:
Total de compras únicas o esporádicas, probablemente compras no recurrentes o de mayor monto.

Installment Purchases:
Total de compras realizadas en cuotas o plazos.

Cash Advance:
El monto total de adelantos en efectivo tomados por el cliente usando su tarjeta de crédito.

Purchases Frequency:
Frecuencia con la que el cliente realiza compras con su tarjeta, probablemente un valor entre 0 y 1.

One-off Purchases Frequency:
Frecuencia de compras esporádicas o únicas.

Purchases Installments Frequency:
Frecuencia de compras realizadas en cuotas o plazos.

Cash Advance Frequency:
Frecuencia con la que el cliente utiliza adelantos en efectivo.

Cash Advance TRX:
Número total de transacciones de adelanto en efectivo realizadas por el cliente.

Purchases TRX:
Número total de transacciones de compra realizadas.

Credit Limit:
Límite de crédito asignado al cliente por el emisor de la tarjeta.

Payments:
El monto total de pagos realizados por el cliente en los últimos 6 meses.

Minimum Payments:
El monto total de pagos mínimos realizados por el cliente, según lo estipulado por el emisor.

PRC Full Payment:
Proporción de pagos realizados en su totalidad respecto al balance total. Puede ser un valor entre 0 y 1.

Tenure:
El tiempo total (en meses) durante el cual el cliente ha sido titular de la tarjeta de crédito."

"Leemos el dataset y obtenemos algunos datos básicos para poder ir estudiandolo
y entendiendolo."
data <- read.csv("market_segmentation_insurance_unsupervised.csv")
head(data)
str(data)
dim(data)
summary(data)

"Eliminamos el id."
data$CUST_ID <- NULL

"Tenemos dos columnas con nulos, sobre todo en la columna de pagos mínimos."
"Comprobamos el porcentaje de nulos en esa columna correspondientemente. Vemos
que solo un 3.5% de los datos totales de esa columna es nulo."
(sum(is.na(data$MINIMUM_PAYMENTS)) / nrow(data)) * 100

"Pese a ser un porcentaje bajo, voy a codificarlos como valor de tipo desconocido.
El nulo de límite de crédito si lo voy a eliminar por ser solo uno, pues considero
que no puede aportar información siendo tan solo un valor faltante.

No elimino los nulos de la columna de pagos mínimos pues creo que pueden aportar
información, aunque no estoy seguro por no tener más conocimiento del problema."
data <- data %>% filter(!is.na(CREDIT_LIMIT)) # Borrado de nulo en Credit limit
data$MINIMUM_PAYMENTS[is.na(data$MINIMUM_PAYMENTS)] <- -1

"Vamos a discretizar los valores de las variables. Para ello creo 3 tipos de
funciones de discretización de valores continuos. Aquellos que tienen valores y
rangos muy amplios son continuos high, valores y rangos medios continuos y valores
y rangos bajos continuos low. Para valores continuos y con valores negativos (-1)
como en el caso de pago mínimo, he añadido la categoría de desconocido, es decir,
los antiguos nulos. El resto son o por frecuencia (entre [0,1]) o por período (
meses)."
discretize_continuous <- function(variable) {
  cut(variable, breaks = c(-Inf, 0, 500, 1000, 5000, Inf), 
      labels = c("Unknown", "Low", "Medium", "High", "Very High"), 
      right = FALSE)
}

discretize_frequency <- function(variable) {
  cut(variable, breaks = c(-Inf, 0.33, 0.66, Inf), 
      labels = c("Low", "Medium", "High"), 
      right = FALSE)
}

discretize_continuous_low <- function(variable) {
  cut(variable, breaks = c(-Inf, 0, 5, 15, Inf), 
      labels = c("None", "Low", "Medium", "High"), 
      right = FALSE)
}

discretize_continuous_high <- function(variable) {
  cut(variable, breaks = c(-Inf, 2000, 5000, 10000, Inf), 
      labels = c("Low", "Medium", "High", "Very High"), 
      right = FALSE)
}

discretize_periods <- function(variable) {
  cut(variable, breaks = c(-Inf, 6, 12, Inf), 
      labels = c("Short-term", "Medium-term", "Long-term"), 
      right = FALSE)
}

data$BALANCE <- discretize_continuous(data$BALANCE)
data$BALANCE_FREQUENCY <- discretize_frequency(data$BALANCE_FREQUENCY)
data$PURCHASES <- discretize_continuous(data$PURCHASES)
data$ONEOFF_PURCHASES <- discretize_continuous(data$ONEOFF_PURCHASES)
data$INSTALLMENTS_PURCHASES <- discretize_continuous(data$INSTALLMENTS_PURCHASES)
data$CASH_ADVANCE <- discretize_continuous(data$CASH_ADVANCE)
data$PURCHASES_FREQUENCY <- discretize_frequency(data$PURCHASES_FREQUENCY)
data$ONEOFF_PURCHASES_FREQUENCY <- discretize_frequency(data$ONEOFF_PURCHASES_FREQUENCY)
data$PURCHASES_INSTALLMENTS_FREQUENCY <- discretize_frequency(data$PURCHASES_INSTALLMENTS_FREQUENCY)
data$CASH_ADVANCE_FREQUENCY <- discretize_frequency(data$CASH_ADVANCE_FREQUENCY)
data$CASH_ADVANCE_TRX <- discretize_continuous_low(data$CASH_ADVANCE_TRX)
data$PURCHASES_TRX <- discretize_continuous_low(data$PURCHASES_TRX)
data$CREDIT_LIMIT <- discretize_continuous_high(data$CREDIT_LIMIT)
data$PAYMENTS <- discretize_continuous(data$PAYMENTS)
data$MINIMUM_PAYMENTS <- discretize_continuous(data$MINIMUM_PAYMENTS)
data$PRC_FULL_PAYMENT <- discretize_frequency(data$PRC_FULL_PAYMENT)
data$TENURE <- discretize_periods(data$TENURE)

summary(data)

"Ahora, para pdoer trabajar con el dataset y las reglas de asociación, debemos
transformarlo a un conjunto de transacciones."

data_trans <- as(data, "transactions")
data_trans
summary(data_trans)

itemFrequencyPlot(data_trans, support = 0.1, cex.names=0.8)
