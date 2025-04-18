- #normaldistribution | #distribuciónnormal

# Introducción
- La distribución normal o distribución Gausiana ( #gaussian ) es una distribución estadística de datos simétrica con forma de campana, cuyo punto [[mean and weighted mean|medio]] es el que más "masa" acumula.
- Su difusión es la que se conoce como [[variance and standard deviation|desviación estándar]].
- Las "colas" de esta distribución se hacen más finas a medida que se alejan de la media.

![[normal-distribution.png]]

# Propiedades
- Simétrica con respecto a la media.
- Su difusión está directamente relacionada con la [[variance and standard deviation|desviación estándar]].
- Las "colas" son valores muy improbables y se acercan infinitamente al cero sin llegar nunca a serlo.
- Se asemeja bastante a muchos problemas cotidianos, incluso a problemas "no normales" gracias al [[central limit theorem|teorema central del límite]].

# Función de densidad de probabilidad
- #pdf
- $f(x)=\frac{1}{\sigma}·\sqrt{2\pi}·e^{-\frac{1}{2}(\frac{x-\mu^2}{\sigma})}$
- Esta función es continua, de forma que para calcular la probabilidad debemos integrar un rango de valores de $x$ para encontrar el área.
- Es una función que describe la probabilidad de que una variable aleatoria caiga dentro de **cierto rango de valores**. En otras palabras, la **PDF** proporciona una descripción de cómo se distribuyen las probabilidades de los posibles valores de una variable aleatoria continua.

# Función de distribución acumulativa
- #cdf
- El eje vertical de la distribución normal mide la [[probability#^ae256f|verosimilitud]] y no la probabilidad de los datos.
- Para comprobar la probabilidad debemos encontrar el área bajo la curva de cierto rango de valores.
- De forma similar a como función la [[beta distribution|distribución beta]], podríamos decir que esta función acumulativa calcula calcula el área hasta un valor dado de $x$ para una distribución dada.
![[cumulative-distribution-function.png]]
- La función da como salida el valor exacto del área hasta un valor $x$. Si en el ejemplo dado anteriormente, calculamos para el valor $0.0$, nos daría un resultado $f(0.0)=0.5$ (debido a la simetría de la distribución normal).

# Función de distribución acumulativa inversa
- Esta función devuelve el valor $x$ correspondiente al área $y$ dado por la $CDF$. Es la inversa de la función $CDF$.
- También puede utilizarse para generar números aleatorios que sigan una districuión normal. Se generan valores aleatorios entre $0.0$ y $1.0$ y se pasan por parámetro a la $CDF^-1$.

# Distribución normal estándar
- [[z-score#Distribución normal estándar|standard normal distribution]]

# Transformaciones
- Si se tiene una distribución normal, es decir:
$$P(x|\mu \varSigma)=N_{x}(\mu, \varSigma)$$
- Si se le aplica una [[linear transformations|transformación lineal]] a la variable $x$, entonces la nueva variable también sigue una distribución normal.