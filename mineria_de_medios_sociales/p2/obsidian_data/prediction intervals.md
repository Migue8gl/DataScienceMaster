- #predictionintervals | #intervalosdepredicciones

- La línea creada con la regresión lineal tiene asociada a ella una [[normal distribution|distribución normal]]. Por ello, cada predicción $y$ es un parámetro estadístico, como la [[mean and weighted mean|media]].
- La [[variance and standard deviation|varianza y desviación estándar]] son conceptos que se aplican también en la regresión lineal.
- La media de nuestra "curva de Bell" sería aquel punto que toque la línea de la regresión, mientras que lo alejados y esparcidos que estén los puntos de esa línea reflejan la varianza/desviación.
- Por ello, para cada valor $y$ predicho, hay un [[confidence intervals|intervalo de confianza]].

![[prediction-interval-example.png|400]]

- Para calcularlo, necesitamos calcular el margen de error y sumarlo/restarlo a la predicción **y**, esto nos dará los intervalos.
- Necesitamos un valor crítico de la [[t-distribution|distribución T]] y [[residual standard error|el error estándar de la estimación]].
	- $E=t_{.025}·S_e·\sqrt{1+\frac{1}{n}+\frac{n(x_0+\bar{x})²}{n(\sum x²)-(\sum x)²}}$
- El valor en el que estamos interesados por conocer los intervalos es $x_0$.