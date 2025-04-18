- #ml | #linearregression | #statistics 

# Concepto
- La regresión lineal es un método de aprendizaje supervisado en el que se asume que hay dependencia de tipo lineal entre la variable dependiente $Y$ y las variables predictoras o independientes $X_i$
- Se asume un modelo:
	- $Y=\beta_0+\beta_{1}\cdot X + \epsilon$
- Donde las $\beta$ son los coeficientes o parámetros del modelo. Son dos variables aprendidas. En el caso de la regresión lineal son el **intercepto** y la **pendiente**.
- La variable $\epsilon$ se refiere al error irreducible del modelo.
- [[linear regression fitters]].

# Métricas
## RSS
- *Residual Sum of Squares* ([[sum of squares|Suma de cuadrados residual]]) -> Es la suma de los [[residuals|residuos]] o errores cometidos por cada punto elevados al cuadrado.
- El método de *mean squared error* intenta minimizar el **RSS**.
## SE
- *Standard Error of an Estimator* (Error estándar de un predictor) -> El error estándar de un estimador refleja cómo varía en un muestreo repetido.
- $SE(\hat{\beta_1})^2 = \frac{\sigma^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \quad$
- $SE(\hat{\beta_0})^2 = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2} \right]$
- Estos errores son útiles a la hora de crear [[confidence intervals|intervalos de confianza]]. Los intervalos para un $95\%$ de confianza para $\beta_1$ son: $\left[ \hat{\beta_1} - 2 \cdot SE(\hat{\beta_1}), \hat{\beta_1} + 2 \cdot SE(\hat{\beta_1}) \right]$.
- Los errores estándar también se pueden utilizar para computar [[hypothesis testing|tests de hipótesis]].
![[linear regression-hypothesis-test.png|500]]
- Cálculo del estadístico $t$:
	- $t = \frac{\hat{\beta_1} - 0}{SE(\hat{\beta_1})}$
- Mide cuántas desviaciones estándar está el coeficiente estimado de cero.
- Valores absolutos más altos indican mayor significancia estadística.
## RSE
- *Residual Standard Error* ([[residual standard error|Error estándar de los residuos]]) -> Es una métrica que cuantifica como de bien ajusta el modelo de regresión a los puntos.
## R² 
- *R-squared* -> Otra métrica que mide la calidad del modelo.
- Si $R^2=1$, significa que el modelo explica el $100\%$ de la variabilidad de los datos; es decir, el modelo tiene un ajuste perfecto.
- Si $R^2=0$, significa que el modelo no explica nada de la variabilidad de los datos, y los valores predichos son simplemente el promedio de los valores observados.
- **Sensibilidad al número de predictores**: Aumentar el número de variables en un modelo puede incrementar el $R^2$ incluso si esas variables no son realmente útiles. Por ello, se suele utilizar el $R^2$ ajustado, que penaliza la adición de predictores irrelevantes.
![[linear regression-r-squared.png]]

![[linear regression-difference-rss-tss.png]]
## Estadístico F
- $F = \frac{(TSS - RSS) / p}{RSS / (n - p - 1)} \sim F_{p,n-p-1}$
- Es una herramienta crucial para evaluar la utilidad general del modelo de regresión, permitiéndonos determinar si al menos uno de los predictores contribuye significativamente a explicar la variabilidad en la variable dependiente.
- Para valores cercanos a uno, se estima que ningún parámetro contribuye. Valores muy superiores a uno indican que hay evidencia de que al menos uno de ellos contribuye significativamente.
![[linear regression-f-statistic-example.png|500]]

# Interpretación de coeficientes de regresión
### Escenario ideal: predictores no correlacionados
- **Diseño equilibrado**: Las variables independientes no están correlacionadas.
- **Estimación y prueba independiente**: Cada coeficiente $\beta_j$ se puede estimar y evaluar por separado.
- **Interpretación clara**: Un cambio de una unidad en $X_j$  se asocia con un cambio de $\beta_j$ en $Y$, manteniendo fijas las demás variables.
### Problemas con predictores correlacionados
- **Mayor varianza en coeficientes**: La [[multicollinearity|multicolinealidad]] aumenta la incertidumbre en las estimaciones.
- **Interpretación complicada**: Cambios en $X_j$ afectan a otras variables, lo que dificulta mantener constante todo lo demás.
### Evitar afirmaciones de causalidad
- **Datos observacionales**: Las asociaciones no implican causalidad; otros factores pueden influir en los resultados.

# Selección de características
- [[feature selection|Selección por métodos estadísticos]].