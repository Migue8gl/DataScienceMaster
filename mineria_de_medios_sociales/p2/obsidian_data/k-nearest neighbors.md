- #knn | #ml

## Introducción
- El algoritmo de k vecinos más cercanos (**knn**) es un conocido clasificador de aprendizaje supervisado, no paramétrico, que utiliza la proximidad para realizar sus predicciones.
- Puede ser utilizado para regresión o clasificación, pero su uso suele darse más en este último problema.
- Trabaja asumiendo que puntos que se encuentren cercanos en el hiper-espacio tienen alta probabilidad de ser del mismo tipo. 
- En problemas de clasificación, se asigna una etiqueta de clase en función de una votación mayoritaria, es decir, se utiliza la etiqueta que está representada con mayor frecuencia alrededor de un punto de datos dado. Aunque esto se considera técnicamente una "votación de pluralidad", el término "votación mayoritaria" se utiliza más comúnmente en la literatura. La distinción entre estas terminologías radica en que la "votación mayoritaria" técnicamente requiere una mayoría superior al $50\%$, lo cual funciona principalmente cuando solo hay dos categorías. Cuando hay múltiples clases, por ejemplo, cuatro categorías, no es necesario tener el $50\%$ de los votos para llegar a una conclusión sobre una clase; se podría asignar una etiqueta de clase con un voto superior al $25\%$.

![[knn-example.png]]

## Distancias
- Algunas de las más usadas (depende del problema y del tipo de datos) son las siguientes métricas de [[distances|distancia]]:
### Distancia Euclídea (p=2)
- Mide una línea recta entre el punto pasado por parámetro y otro punto, está limitada a valores [[number theory|reales]].
- Fórmula -> $d(x,y)=\sqrt{\sum_{i=1}^n(y_i-x_i)²}$

### Distancia Manhattan (p=1)
- Mide el valor absoluto entre dos puntos. Se suele visualizar con una cuadrícula que ilustra cómo se puede ir de una dirección a otra por las calles de la ciudad.
- Fórmula -> $d(x,y)=(\sum_{i=1}^n|x_i-y_i|)$

### Distancia Minkowski
- Es la generalización de las distancias **Manhattan** y **Euclídea**. El parámetro $p$ permite la creación de otro tipo de distancias. Cuando $p=2$ tenemos la distancia **Euclídea**, mientras que con $p=1$ la **Manhattan**.
- Fórmula -> $d(x,y)=(\sum_{i=1}^n|x_i-y_i|)^{\frac{1}{p}}$

### Distancia Hamming
- Este tipo de distancia se utiliza al manejar [[vector|vectores]] de *strings* o booleanos, identificando los puntos donde los vectores no son iguales. Define la distancia por el coeficiente de solapamiento entre vectores, a más distintos mayor distancia.
- Fórmula -> $D_H=(\sum_{i=1}^n|x_i-y_i|)$; $x=y \implies D=0$; $x\neq y \implies D=1$

![[distances-functions-example.png|500]]

## Valor de K
- Valores mayores de $k$ incrementan la [[overfitting and variance|varianza]], pero decrementan el sesgo (**bias**). De forma inversa, a menor $k$ mayor sesgo y menor varianza. 
- Otra forma de verlo es que si $k=1$ (el valor mínimo), el modelo entrenado será muy complejo, pues cada punto puede variar de clase por mínimo que sea el cambia en su entorno. Es un modelo que tiende a sobreajustar. Con $k=n$ el modelo es lo más simple posible, al tener en cuenta absolutamente todos los puntos, la clasificación será siempre la de la clase más repetida.
- Se recomienda usar un valor impar para evitar empates en la decisión de clasificación.

## Características
- Es fácil de implementar, fácil de adaptar y sin parámetros (quitando $k$ y la métrica de distancia).
- El algoritmo knn suele caer en [[curse of dimensionality|la maldición de la dimensionalidad]] con facilidad.
- Tiende al sobreajuste pues es muy delicado a outliers. Además es muy sensible al escalado de características.