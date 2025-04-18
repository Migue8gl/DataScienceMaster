- #machinelearning | #missingvalues

# Suposiciones y mecanismos
- Se tiene $X=(X_{obs}, X_{mis})$.
- Se dice que $X_{obs}$ es la parte observada de $X$ y $X_{mis}$ la parte de valores perdidos.
- Se supone que se tiene una matriz $B$ del mismo tamaño de $X$ donde aquellas celdas con $0$ significan valores perdidos y aquellas con $1$ valores no perdidos, de los que se tiene información.
- La distribución de $B$ debería estar relacionada con $X$ y con algunos parámetros desconocidos $\zeta$, así que disponemos de un modelo de probabilidad para $B$ descrito por $P(B|X, \zeta)$.
## MAR
- Si tenemos suposiciones *missing at random*, significa que la distribución de $B$ no depende de $X_{mis}$. Por tanto depende de un error desconocido y de los propios valores observados -> $P(B | X_{obs}, X_{mis}, \zeta) = P(B | X_{obs}, \zeta).$
## MCAR
- *Missing completely at random* es un caso especial de **MAR**, en la que los datos perdidos no dependen de los datos observados tampoco, sino de otros factores como el azar -> $P(B | X_{obs}, X_{mis}, \zeta) = P(B |\zeta).$
- Bajo este caso, es viable eliminar aquellos elementos con variables faltantes (si no son muchas) sin perder demasiada información.
## MNAR
- Ocurre cuando **MAR** no es aplicable, pues la falta de datos depende tanto de los datos no faltantes como de los propios valores perdidos. Esto se conoce como *missing not at random*.
- La única manera de obtener un estimador fiable es modelar la propia manera de perder los datos.
- Esta es una tarea compleja que implica crear un modelo para lidiar con los valores perdidos. Luego, ese modelo se incorporaría a otro modelo aún más complejo que se utilizaría para estimar los valores faltantes.

Los pasos clave serían:
1. Desarrollar un modelo inicial para manejar los valores perdidos.
2. Incorporar ese modelo de valores perdidos en un modelo de predicción más grande.
3. Utilizar el modelo de predicción completo para estimar los valores faltantes en los datos.

# Formas de imputación
- [[imputations]]