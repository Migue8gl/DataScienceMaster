#ml | #overfitting | #regularization

## Introducción
- El sobreajuste o [[overfitting and variance|overfitting]] es un problema muy común en aprendizaje automático. Ocurre cuando un modelo ajusta muy bien los datos de entrenamiento, pero no generaliza adecuadamente en datos nunca vistos.
- Cuando un modelo sufre de sobreajuste, se suele decir que tiene una varianza muy alta, mientras que si sufre de *underfitting* entonces el sesgo es alto.
- Una forma de paliar en cierto grado el sobreajuste es con la **regularización**.
- Es un método adecuado para manejar [[collinearity|colinealidad]], filtrando el ruido de los datos y previniendo el sobreajuste.
- El concepto principal detrás de la regularización consiste en añadir información para penalizar valores extremos de los pesos.
-  Para que la regularización funcione correctamente, el normalizado o escalado de los datos es importante.

## Tipos
### Regularización L2 o weight decay
- Consiste en:
	- $\frac{\lambda}{2n}\vert\vert w\vert\vert^2 = \frac{\lambda}{2n}\sum_{j=1}^m w_j^2$
- El parámetro $\lambda$ es el llamado parámetro regularizador. El $2$ es meramente un factor de escalado que se cancela cuando se computa el gradiente y $n$ se añade para escalar el término de la regularización a la pérdida.
- Se suma a la función de pérdida para poder usarlo.
- Incrementando el valor de $\lambda$ se incrementa la fuerza regularizadora.
- El sesgo, que suele ser el incercepto o simplemente un umbral negativo, no se regulariza.