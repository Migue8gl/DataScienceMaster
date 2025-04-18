- #statistics | #covariance

# Concepto
- La covarianza mide el grado de relación lineal entre dos variables. Si tenemos dos variables $X$ y $Y$ y queremos analizar cómo varían conjuntamente, la covarianza nos ayudará a saber si sus variaciones son en la misma dirección (covarianza positiva) o en direcciones opuestas (covarianza negativa).
$$\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y)$$
- La covarianza depende de la escala de las variables, lo que puede hacer difícil comparar covarianzas de diferentes pares de variables. Por esta razón, es común normalizarla al calcular el **coeficiente de correlación**.
- Para una misma matriz, la matriz de covarianza de $X$ es:
$$\frac{1}{N}X^TX-\mu_X$$
- Si no se le resta la media y no se divide por el número de observaciones, es una **matriz de dispersión**.
- Si se estadariza con media cero, no hace falta restarle la media.
- Si la matriz de covarianza es diagonal, significa que las variables son [[linear dependence|independientes]] de manera lineal.