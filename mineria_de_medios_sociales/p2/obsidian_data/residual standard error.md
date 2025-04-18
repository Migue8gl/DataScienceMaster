- #residualstandarderror | #rse

- Es una medida de la dispersión de los [[residuals|residuos]] (errores) en un modelo de regresión lineal. Los residuos son las diferencias entre los valores observados y los valores predichos por el modelo.
- Podemos calcular el **RSS** como se indica en [[sum of squares|esta fórmula]]. Si queremos escalar el error a la unidades de medida originales, para un análisis más sencillo, podemos tomar la raíz del **RSS**. Después tomaremos la media de ese cálculo, y eso, es lo que se conoce como *error estándar de los residuos*:
![[residual standard error-example.png]]
- En la regresión lineal, al tener dos variables, tenemos $n-(2+1)$ [[degrees of freedom|grados de libertad]]. $k$ es el número de parámetros estudiados.
- En resumen, esta métrica dice aproximadamente como de grandes son los errores ([[residuals|residuos]]) para tu conjunto de datos en las mismas unidades que $Y$.