- #hypothesistesting

- Cuando hacemos test sobre hipótesis, tenemos dos principales formas de hacerlo. Podemos optar por los **one-tailed test** (test de una sola cola) o los **two-tailed test**
- La fuerza estadística de los tests solo se consigue cuando se rechaza, pues sabemos que un caso dado es casi imposible por probabilidad. Por ello, no rechazarla lo único que nos dice es que **podría o no podría** ser, pero que no se tiene evidencia de una cosa o la otra.

# Hipótesis teoría
- $H_0$ (nula) -> es una afirmación que representa el estado actual de las cosas o ausencia de efecto. Se quiere refutar o rechazar.
- $H_1$ (alternativa) -> es la afirmación contraria a la hipótesis nula. Representa la existencia de un efecto o diferencia.

### One-tailed test
- Cuando abordamos este tipo de test, lo hacemos usando inecuaciones. 
- Se estudia si nuestro parámetro poblacional a estudiar es mayor o igual a cierto número o si por el contrario, es menor.
		- $H_0:\text{parameter tested}\ge x$
		- $H_1:\text{parameter tested}< x$
- **Hipótesis Nula $H_0$** -> Hipótesis que afirma que no existe relación entre dos conjuntos de datos o variables analizadas. La hipótesis nula es que cualquier diferencia observada experimentalmente se debe únicamente al azar, y que no existe una relación causal subyacente, de ahí el término "*nula*". Además de la hipótesis nula, también se desarrolla una hipótesis alternativa, que afirma que sí existe una relación entre dos variables ($H_1$).
- Para poder rechazar nuestra hipótesis nula, y por tanto, poder afirmar que la relación entre casos no se ha dado por el azar, entonces el caso en cuestión debe tener una probabilidad igual o menor al [[p-values|p-value]].
- Por lo general se acepta un **límite** de significancia del p-value del $5\%$ o lo que es lo mismo, $t=0.05$.
- Si el **p-value** calculado (podemos usar [[normal distribution#Función de distribución acumulativa|la función CDF]] para calcular la probabilidad dado el p-value) es mayor estricto que el límite, fallamos en rechazar la hipótesis nula, es decir, no podemos afirmar con suficiente seguridad que el caso se haya debido al azar, no podemos asegurar que dos variables estén relacionadas de algún modo.

![[one-tailed-test-image.png]]

### Two-tailed test
- A diferencia de los test de una cola, en este tipo de test trabajamos con igualdades.
	- $H_0:\text{parameter tested}= x$
	- $H_1:\text{parameter tested}\neq x$
- Esto es igual a mirar el otro extremo de la distribución, tener en cuenta la segunda cola y con ello tener en cuenta todas las diferencias que pueda haber.
- El test de una sola cola mira si hay diferencias, pero en una dirección elegida (si el grupo A mejora con respecto al grupo B, si el grupo A tiene menos disposición a algo que el grupo B etc.). En el caso del test de dos colas, podríamos tener los siguientes ejemplos (si el grupo A mejora o empeora respecto al grupo B, si el grupo A tiene más disposición que el grupo B, o menos).
- Si cogemos un límite de significancia estadística del $5\%$, entonces, al estar testeando ambos extremos de la distribución, deberemos dividir ese porcentaje en dos, $2.5\%$ para cada uno.
- Este tipo de hipótesis es más restrictiva, por lo que para pasar los test se necesita más evidencia, es más **segura**.
 
![[two-tailed-test-image.png]]