library(arules)
library(tidyverse)
library (arulesViz)

"El dataset ha sido obtenido de Kaggle."
"https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised"

"Variables explicadas en:
https://medium.com/@cyrilladrian.wicaksono/market-segmentation-in-insurance-an-unsupervised-learning-approach-c3b4bd1f309e"

"Descripción de las columnas:

Balance:
El saldo actual del cliente en la tarjeta de crédito.

Balance Frequency:
Frecuencia con la que el cliente mantiene un saldo en su tarjeta de crédito. Podría ser un valor entre 0 y 1, donde valores más altos indican un uso más consistente.

Purchases:
El monto total de compras realizadas por el cliente.

One-off Purchases:
Total de compras únicas o esporádicas, probablemente compras no recurrentes o de mayor monto.

Installment Purchases:
Valor total de compras realizadas en cuotas o plazos.

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
data$ONEOFF_PURCHASES_FREQUENCY <- NULL
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
    labels = c("Unknown", "Low", "Medium", "High"),
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
    breaks = c(-Inf, 7, 12, Inf),
    labels = c("Short-term", "Medium-term", "Long-term"),
    right = FALSE
  )
}

discretize_frequency <- function(variable) {
  cut(
    variable,
    breaks = c(-Inf, 0, 500, 1000, 5000, Inf),
    labels = c("Unknown", "Low", "Medium", "High", "Very High"),
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

"Ahora, para poder trabajar con el dataset y las reglas de asociación, debemos
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
tamaño 4-5 son muy frecuentes en el dataset."
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

"Filtramos para eliminar compras esporádicas altas y pagos altos, ya que suelen
estar también muy relacionados, aunque no sean exactamente lo mismo"
rules_sorted <- head(sort(subset(
  rules,
  subset = !(lhs %in% "ONEOFF_PURCHASES=High") &
    !(lhs %in% "PAYMENTS=High") &
    lift > 1.5
), by = "lift"), 20)

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_pruned <- rules_sorted[!redundant]
inspect(rules_pruned)

"LA regla con mayor lift es:
          {INSTALLMENTS_PURCHASES=Low,
          MINIMUM_PAYMENTS=Medium,
          TENURE=Long-term}            => {BALANCE=High}.
Si un cliente tiene un monto total de pagos en plazos bajo, con un pago mínimo medio
y una posesión de la tarjeta de crédito muy larga, es probable que el balance de su
tarjeta sea alto. Un balance alto se refiere a una cantidad elevada de deuda o saldo pendiente.
Pagos mínimos medianos sugieren que, aunque se están haciendo pagos, estos no
son lo suficientemente grandes como para reducir el saldo rápidamente. Esto
permite que los intereses se acumulen y el saldo permanezca alto.
Un plazo largo significa que la persona está tomando tiempo para pagar la deuda,
lo que contribuye a que el saldo permanezca alto durante más tiempo y siga
acumulando intereses. Pese a que el valor total de compras financiadas a plazos
sea bajo o nulo, es posible que este tipo de personas que prefieren realizar con
poca frecuencia este tipo de compras, tiendan a acumular más deuda.

De hecho, la regla:
        {INSTALLMENTS_PURCHASES=Low,
        MINIMUM_PAYMENTS=Medium}      => {BALANCE=High}
Indica que solo con esas dos variables se suele tener un valor alto de balance. La
regla:
        {MINIMUM_PAYMENTS=Medium}     => {BALANCE=High}
Indica que solo con tener pagos mínimos de valor medio se tiene un balance alto. Esto
quizá indica que esta variable tiene mucho peso sobre la variable de balance, por
ello reglas con más items no empeoran el valor de lift y confianza.
"

"Vamos a analizar las reglas que solo tengan en la parte derecha un balance bajo."
rules_sorted <- head(sort(subset(
  rules,
  subset = !(lhs %in% "ONEOFF_PURCHASES=High") &
    !(lhs %in% "PAYMENTS=High") &
    (rhs %in% "BALANCE=Low") &
    lift > 1.5
), by = "lift"), 20)

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_pruned <- rules_sorted[!redundant]
inspect(rules_pruned)

"La regla con menos items, ya que el resto de reglas la contienen y tienen valores
parecidos de confianza y lift, es:
    {CASH_ADVANCE=Low,
     PURCHASES_TRX=Medium,
     MINIMUM_PAYMENTS=Low}  => {BALANCE=Low}
Esta regla relaciona una cantidad total de retirada de dinero físico de la
tarjeta de crédito baja, una cantidad de compras por transacciones bajas y un pago
mínimo bajo con un balance más sano. Esto hace ver que la variable de pagos mínimos
no es tan importante como anteriormente se analizó, ya que en combinación con ciertos
hábitos con la tarjeta de crédito, es posible disminuir esta deuda incluso pagando
un mínimo bajo."

"Analicemos los consecuentes con pagos mínimos bajos. Lo que encontramos es que normalmente
si se tiene un balance (deuda) bajo, los pagos mínimos son muy bajos"
rules_sorted <- head(sort(subset(
  rules, subset = (rhs %in% "MINIMUM_PAYMENTS=Low") &
    lift > 1.5
), by = "lift"), 20)

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_pruned <- rules_sorted[!redundant]
inspect(rules_pruned)

unique_rhs <- unique(rhs(rules))
inspect(unique_rhs)

"Ahora vamos a analizar grupos que considero interesantes, generando nuevas reglas
a partir de estos."

"Analizamos clientes que retiren grandes cantidades de dinero en metálico usando
la tarjeta de crédito."
rules_cash_advance <- apriori(
  data_trans,
  parameter = list(
    support = 0.1,
    confidence = 0.7,
    minlen = 2
  ),
  appearance = list(lhs = "CASH_ADVANCE=High")
)

rules_cash_advance_sorted <- sort(rules_cash_advance, by = "lift")
inspect(head(rules_cash_advance_sorted))

"Obtenemos con una confianza del 70% y un lift de 1.25, que si los clientes que retiran
grandes cantidades de dinero suelen gastar poco usando la tarjeta de crédito, quizás
por que prefieren el metálico.

Ocurre lo mismo con las compras a plazos usando la tarjeta."

rules_purchases <- apriori(
  data_trans,
  parameter = list(
    support = 0.1,
    confidence = 0.7,
    minlen = 2
  ),
  appearance = list(lhs = "PURCHASES_TRX=High")
)

rules_purchases_sorted <- sort(rules_purchases, by = "lift")
inspect(head(rules_purchases_sorted))

"Ahora vamos a intentar obtener reglas que tengan MINIMUM_PAYMETS con valor
desconocido, quizá se pueda obtener información interesante sobre el por qué
de tantos nulos en esa variable."
rules_minimum_unknown <- apriori(
  data_trans,
  parameter = list(
    support = 0.0001,
    confidence = 0.3,
    minlen = 2
  ),
  appearance = list(lhs = "MINIMUM_PAYMENTS=Unknown")
)

rules_rules_minimum_unknown_sorted <- sort(rules_minimum_unknown, by = "lift")
inspect(head(rules_rules_minimum_unknown_sorted, 10))
plot(rules_rules_minimum_unknown_sorted, method = "graph")

rules_minimum_unknown <- apriori(
  data_trans,
  parameter = list(
    support = 0.0001,
    confidence = 0.3,
    minlen = 2
  ),
  appearance = list(rhs = "MINIMUM_PAYMENTS=Unknown")
)
total_transactions <- length(data_trans)

rules_minimum_unknown_filtered <- subset(rules_minimum_unknown,
                                         subset = (support * total_transactions) >= 10)

rules_minimum_unknown_sorted <- sort(rules_minimum_unknown_filtered, by = "lift")
inspect(head(rules_minimum_unknown_sorted, 50))
plot(rules_minimum_unknown_sorted[1:5], method = "graph")

"Cuando los pagos mínimos son desconocidos, los pagos hechos con la tarjeta son bajos.
Aunque es rara, cuando ocurre, la relación entre las premisas y la conclusión
es fuerte, como lo indica el valor de lift. Además tiene una confianza del 90%.
El resto de reglas obtenidas relaciona todos los casos de valores faltantes con
valores bajos en las variables. Esto puede ser clientes que no utilicen la tarjeta
de crédito, por lo cual tendría sentido para explicar tanto los valores bajos como los valores
faltantes."

"Añadimos los negados de los dos items que más se repiten en las transacciones.
La negación en este caso, al tratarse de una variable continua discretizada, agrupa
los intervalos, haciendo que no se centre en intervalos como HIGH (por ejemplo) y lo haga
en LOW y MEDIUM a la vez."
summary(data_trans)

data_neg <- data %>%
  mutate(
    PRC_FULL_PAYMENT_NOT_HIGH = PRC_FULL_PAYMENT != "High",
    PRC_FULL_PAYMENT_NOT_MEDIUM = PRC_FULL_PAYMENT != "Medium",
    PRC_FULL_PAYMENT_NOT_LOW = PRC_FULL_PAYMENT != "Low",
    TENURE_NOT_LONG_TERM = TENURE != "Long-term",
    TENURE_NOT_MEDIUM_TERM = TENURE != "Medium-term",
    TENURE_NOT_SHORT_TERM = TENURE != "Short-term",
  )
data_trans_neg <- as(data_neg, "transactions")
data_trans_neg
summary(data_trans_neg)

rules_neg <- apriori(data_trans_neg,
                     parameter = list(
                       support = 0.01,
                       confidence = 0.6,
                       minlen = 2,
                       maxlen = 4
                     ))

"Limitamos el número máximo de items, así quizá encontramos reglas más interpretables."

summary(rules_neg)
rules_sorted <- sort(subset(
  rules_neg,
  subset = (lhs %in% "TENURE_NOT_LONG_TERM") &
    !(rhs %in% "TENURE=Medium-term"),
  lift > 1.2
), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 80))

rules_sorted <- sort(subset(rules_neg, subset = (lhs %in% "TENURE_NOT_MEDIUM_TERM"), lift > 1.2), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 80))

rules_sorted <- sort(subset(rules_neg, subset = (lhs %in% "TENURE_NOT_SHORT_TERM"), lift > 1.2), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 80))

rules_sorted <- sort(subset(
  rules_neg,
  subset = (lhs %in% "PRC_FULL_PAYMENT_NOT_HIGH"),
  lift > 1.2
), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 50))

rules_sorted <- sort(subset(
  rules_neg,
  subset = (lhs %in% "PRC_FULL_PAYMENT_NOT_MEDIUM"),
  lift > 1.2
), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 50))

rules_sorted <- sort(subset(
  rules_neg,
  subset = (lhs %in% "PRC_FULL_PAYMENT_NOT_LOW"),
  lift > 1.1
), by = "lift")

redundant <- is.redundant(x = rules_sorted, measure = "confidence")
rules_sorted <- rules_sorted[!redundant]
inspect(head(rules_sorted, 50))

"Algunas de las nuevas reglas son muy redundantes y no aportan información. Es obvio
que si TENURE no es de larga duración, entonces será o baja o media. Muchas de
estas nuevas reglas aportan ese tipo de información."

"Se extraen, de todas formas, algunas reglas interesante:
 {BALANCE=High, CREDIT_LIMIT=Medium, TENURE_NOT_LONG_TERM} => {CASH_ADVANCE=High}
Clientes con saldo alto, límite de crédito medio y tenencia corta
o media tienden a usar adelantos en efectivo en gran medida.
{PURCHASES_TRX=Low, MINIMUM_PAYMENTS=Medium, TENURE_NOT_LONG_TERM} => {BALANCE=High}
Clientes que no realizan muchas compras con transacciones, hacen pagos mínimos de
tamaño medio y que no llevan mucho tiempo con la tarjeta, suelen tener una deuda
alta.
"

"En cuanto a PRC_FULL_PAYMENT_NOT_HIGH, la mayoría de reglas van asociadas a saldos
(deudas) muy altas y pagos y compras muy elevadas, es decir, clientes que gastan y
se endeudan mucho. Teniendo en cuenta que PRC_FULL_PAYMENT_NOT_HIGH significa
que el cliente no tiene una proporción de pagos de deuda altos, podría significar
que son clientes con bastantes riesgo de impago."

"Una de ellas, muy interesante es:
{BALANCE=Very High, PRC_FULL_PAYMENT_NOT_HIGH} => {MINIMUM_PAYMENTS=High}
Interpretación: Clientes con saldo muy alto y bajo porcentaje de pago completo
tienden a realizar pagos mínimos elevados. Esto puede ser como contramedida por los
incumplimientos."
