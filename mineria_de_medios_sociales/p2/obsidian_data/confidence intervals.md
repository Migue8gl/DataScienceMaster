- #confidenceintervals

### Explicación
- Un intervalo de confianza es un rango que muestra como de seguro es que la [[mean and weighted mean|media muestral]] (o cualquier otro parámetro) se asemeje a la realidad de la [[populations|población]].
- Se suele expresar como un porcentaje de seguridad sobre un rango delimitado entre dos valores.

### Nivel de confianza (LOC)
- Se comienza eligiendo un **LOC** (level of confidence), el cual contiene la probabilidad de que el parámetro elegido caiga dentro del rango calculado, puede ser.
- Podemos aprovechar el [[central limit theorem|teorema del límite central]] para inferir el rango para la población.
- Primero, deberemos calcular el **valor crítico z**. Este es el que conforma el rango simétrico dentro de una [[normal distribution#Distribución normal estándar|distribución normal estándar]] para un valor **LOC** de $95\%$.
- Podemos utilizar la [[normal distribution#Función de distribución acumulativa inversa|CDF inversa]]. Para obtener el $95\%$ del área simétrica en el centro, primero deberemos recortar el $5\%$ sobrante de las colas. Usando la función inversa obtenemos los valores de $x$ para las áreas de las colas, que son aquellas con probabilidad $0.025$ y $0.975$.
- Después devolvemos los valores $z$ correspondientes a los extremos. El el caso del $95\%$, el valor crítico de $z$ es de $\pm 1.95996$. Es el valor crítico $z$ que captura el $95\%$ de probabilidad en el centro de una distribución normal estándar.

### Error marginal
- Para calcular el error marginal dado nuestro valor $z$, debemos utilizar la siguiente fórmula.
	- $E=\pm z_c\frac{s}{\sqrt{n}}$
	- $n=$ número de individuos en la muestra, $s=$ [[variance and standard deviation|desviación estándar]].
- Con el error ya calculado, podemos obtener el intervalor de confianza. Este se crea aplicando el error marginal a la [[mean and weighted mean|media muestral]]:
	- Intervalo de confianza del $95\%$ = $\bar{x}\pm E$.

### Interpretación
- Basándonos en una muestra de $n$ individuos con una media muestral de $\bar{x}$ y una desviación estándar de $s$, estoy seguro al $95\%$ de que la media poblacional se encuentra entre $\bar{x}+E$ y $\bar{x}-E$.

- Todo esto funciona si se hace sobre una muestra que siga una distribución normal, es decir, debe haber al menos $31$ individuos, lo cual se dice en el [[central limit theorem|teorema del límite central]].
- Si queremos aplicar un intervalo de confianza en una muestra más pequeña, necesitamos usar una distribución con una varianza más alta (colas más gordas, las cuales reflejan mas incertidumbre). 
- Para ello, se utiliza la [[t-distribution|distribución T]].