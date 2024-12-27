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
El monto total de compras realizadas por el cliente..

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
El monto total de pagos realizados por el cliente.

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

"Eliminamos Balance Frequency, Purchases Frecuency, One-off Purchases Frequency,
Purchases Installments Frequency y Cash Advance Frequency ya que son variables
redundantes que explican lo mismo que las originales, pero midiendolo en frecuencia."
data$BALANCE_FREQUENCY <- NULL
data$PURCHASES_FREQUENCY <- NULL
data$ONEOFF_PURCHASES_FREQUENCY<- NULL
data$PURCHASES_INSTALLMENTS_FREQUENCY <- NULL
data$CASH_ADVANCE_FREQUENCY <- NULL

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
los antiguos nulos. El resto son por período (meses)."
discretize_continuous <- function(variable) {
  cut(
    variable,
    breaks = c(-Inf, 0, 500, 1000, 5000, Inf),
    labels = c("Unknown", "Low", "Medium", "High", "Very High"),
    right = FALSE
  )
}

discretize_continuous_low <- function(variable) {
  cut(
    variable,
    breaks = c(-Inf, 0, 5, 15, Inf),
    labels = c("None", "Low", "Medium", "High"),
    right = FALSE
  )
}

discretize_continuous_high <- function(variable) {
  cut(
    variable,
    breaks = c(-Inf, 2000, 5000, 10000, Inf),
    labels = c("Low", "Medium", "High", "Very High"),
    right = FALSE
  )
}

discretize_periods <- function(variable) {
  cut(
    variable,
    breaks = c(-Inf, 6, 12, Inf),
    labels = c("Short-term", "Medium-term", "Long-term"),
    right = FALSE
  )
}

data$BALANCE <- discretize_continuous(data$BALANCE)
data$PURCHASES <- discretize_continuous(data$PURCHASES)
data$ONEOFF_PURCHASES <- discretize_continuous(data$ONEOFF_PURCHASES)
data$INSTALLMENTS_PURCHASES <- discretize_continuous(data$INSTALLMENTS_PURCHASES)
data$CASH_ADVANCE <- discretize_continuous(data$CASH_ADVANCE)
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

"Mostramos un gráfico de frecuencias de items en el dataset. Los items que parecen
aparecer con mayor frecuencia son:
  - ternure_long_term, lo que indicaría que la mayoría
  de observaciones en el dataset tienen la tarjeta de crédito como titulares desde
  hace mucho tiempo.
  - pcr_full_payment_low, que sugiere que el cliente realiza
  pagos parciales o mínimos, lo que representaría mayor riesgo financiero.
  - cash_advance_frecuency_low, que se refiere a la cantidad de veces que un
  cliente utiliza su tarjeta de crédito para un adelanto de efectivo. Es una tasa
  baja, lo cuál es menos riesgoso, ya que suele estar sujeto a tasas de interés.
  - balance_frecuency_high, que indica que el cliente suele mantener el saldo en
  su tarjeta de forma consistente.

  El soporte alto de estos items muestra que son muy frecuentes en el dataset.
"
itemFrequencyPlot(data_trans, support = 0.1, cex.names = 0.6)

"Vamos a extraer reglas frecuentes. Las reglas de tamaño 1 deberían coincidir
con lo que el diagrama de barras muestra."
i_data <- apriori(data_trans, parameter = list(support = 0.1, target = "frequent"))
i_data <- sort(i_data, by = "support")
inspect(head(i_data, 10))

"Vamos a comprobar qué tipo de reglas (si de tamaño 1, 2, 3 o más) son más
frecuentes en nuestro dataset con la función size. Parece ser que los itemsets de
tamaño 4-6 son muy frecuentes en el dataset."
ggplot(as.data.frame(table(size(i_data))), aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity") +
  xlab("Itemset Size") +
  ylab("Count") +
  theme_minimal()

"Vamos a extraer reglas interesantes según ciertas métricas."
rules <- apriori(data_trans, parameter = list(
  support = 0.1,
  confidence = 0.8,
  minlen = 2
))

"Las reglas obtenidas tiene un lift de entre [0.9, 3.7]. Esto indica que de la
mediana hacia abajo, las reglas obtenidas no relacionan el antecedente con el consecuente,
ya que si lift es ~= 1, indica que la relación no existe. Si es mayor o inferior a 1
entonces se hablarían de relaciones positivas o negativas, pero de la mediana hacia
abajo el valor para esta métrica es casi 1."
summary(rules)

"Vamos a inspeccionar aquellas en orden descendente de lift para ver aquellas que
tienen un valor mayor en esta métrica."
inspect(head(sort(rules, by = "lift"), 20))

"Todas tienen como consecuente puchases_high, es decir, un nivel de gasto con la
tarjeta de crédito muy alto."

"La primera regla, con el lift más alto, es la siguiente:
{ONEOFF_PURCHASES = High, PRC_FULL_PAYMENT = Low} => {PURCHASES = High}
La regla nos dice que si un cliente ha realizado compras de tipo único
(ONEOFF_PURCHASES) de alto valor y ha hecho pagos bajos o incompletos
(PRC_FULL_PAYMENT = Low), entonces es probable que el cliente también tenga un
monto alto de compras generales (PURCHASES = High).
De todas formas, no considero muy interesante esta regla, ya que si el cliente
ha tenido un número de compras no recurrentes altas, es normal que el número de
compras sea alto, es algo lógico. Por ello voy a intentar indagar para obtener
reglas más interesantes y menos obvias."

"Filtramos para eliminar compras esporádicas altas y"
inspect(head(subset(
  rules, subset = !(lhs %in% "ONEOFF_PURCHASES=High") &
    lift > 2.8
), 20))
