- #ensemble | #boosting | #ml

# Concepto
- Se crean modelos predictos de forma secuencial. Cada predictor usa información del predictor anterior.
- Se asigna un peso a cada caso de datos, lo que permite al modelo poner más o menos énfasis en cada observación. Inicialmente, todos los datos pueden tener el mismo peso, pero conforme el modelo progresa, los pesos se ajustan.
- Las observaciones que son clasificadas incorrectamente en una ronda reciben un peso más alto. Esto hace que el algoritmo ponga más atención en estas observaciones en las siguientes iteraciones, con el objetivo de corregir los errores que otros modelos previos cometieron.
- En cada iteración (o "ronda de *boosting*"), el modelo entrena un nuevo clasificador simple (por ejemplo, un [[decision trees|árbol de decisión]] pequeño) utilizando los datos ponderados. Este clasificador intenta mejorar el rendimiento en las observaciones que recibieron un peso más alto en la ronda anterior.
![[boosting-example.png|550]]
- Es una consecución de modelos especializados. Cada modelo intenta especializarse en aquello en lo que que el modelo anterior fallaba.
![[boosting-algorithm.png|550]]

# Hiperparámetros
- **Número de predictores** -> Al contrario que [[bagging]], un número alto de predictores ($B$) puede dar lugar a [[overfitting and variance|sobreajuste]]. Aunque si ocurre suele ocurrir muy lentamente.
- **Parámetro de encogimiento** -> El parámetro de encogimiento $\lambda$ es un número positivo pequeño. Controla el ratio en el que *boosting* aprende. Cada árbol contribuye solo una fracción de su predicción, controlada por $\lambda$. Esto hace que el modelo aprenda más lentamente, ayudando a reducir el riesgo de sobreajuste.
- **Número de divisiones** -> Número de divisiones que cada árbol del *boosting* va a realizar. Es equivalente a la altura de un árbol.

# Clasificación
- Para clasificar se puede hacer una **votación ponderada** de cada árbol o un **muestreo ponderado**.

# State Of The Art
## AdaBoost
 ![[boosting-adaboost.png|500]]
 - La distribución inicial se asume uniforme, por lo que cada dato en el espacio tiene la misma importancia.
 - Despues de entrena a un predictor *débil* el cual produce una hipótesis $h_i$. Esta hipótesis genera una frontera de decisión en el espacio. 
 - Se computa un valor de confianza $\alpha_i$. A mejor sea la hipótesis generada, mayor la confianza.
 - La distribución es actualizada en base a los errores generados por la anterior hipótesis. De esta forma se le da mayor pesos a aquellos datos más complicados. En la siguiente ronda de *boosting* se le dará mayor énfasis aquellos mayormente ponderados.
 - El predictor final es una votación ponderada de las hipótesis generadas por cada predictor débil.
![[boosting-example-final-h.png|500]]
## XGBoost
## LightGBM
## Gradient Boosting