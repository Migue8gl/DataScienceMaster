- #mh | #bat

## Inspiración
- El algoritmo [[metaheuristic|metaheurístico]] de los murciélagos se basa en la capacidad de ecolocalización de estos para detectar a su presa y cazarla finalmente.
- La ecolocalización de los murciélagos no solo les permite detectar a su presa, sino discriminar entre distintos tipos de insectos.
- Más concretamente, este algoritmo se basa en los micro murciélagos, pues estos usan más frecuentemente la ecolocalización en comparación a otras especies de murciélagos.
- Estos murciélagos emiten un pulso de sonido, el cual, rebota en los objetos y entidades cercanas, permitiendo al murciélago componer una "visión" general de sus alrededores. La frecuencia del pulso emitido podría estar relacionada con distintas estrategias de caza.

## Acústica de la ecolocalización
- Aunque cada pulso dura solo unos pocos milisegundos (entre $8$ y $10$ ms), mantiene una frecuencia constante, generalmente entre $25$ kHz y $150$ kHz. La longitud de onda ($\lambda$) de estos impulsos ultrasónicos, con una frecuencia ($f$) y velocidad del sonido ($v$) en el aire a $340$ m/s, está dada por:
	- $\lambda = \frac{v}{f}$
- Con longitudes entre $2$ mm y $14$ mm para frecuencias de $25$ kHz a $150$ kHz, respectivamente. Esta longitud de onda coincide con el tamaño de las presas. Los murciélagos pueden detectar obstáculos tan pequeños como cabellos humanos, utilizando la eco-localización para construir un modelo 3D del entorno, y discriminando distancias, orientaciones y tipos de presas. Aunque los murciélagos tienen buena vista y olfato, utilizan principalmente la eco-localización para la detección de presas y la navegación eficiente. 

## Operadores
Para simplificar, se usan las siguientes reglas aproximadas o idealizadas: 
- Todos los murciélagos utilizan la ecolocación para percibir la distancia, y también conocen la diferencia entre alimento/presa y barreras de fondo de alguna manera mágica. 
- Los murciélagos vuelan aleatoriamente con una velocidad $v_i$ en la posición $x_i$ con una frecuencia fija $f_{min}$, variando la longitud de onda $\lambda$ y la intensidad del sonido $A_0$ para buscar presas. Pueden ajustar automáticamente la longitud de onda (o frecuencia) de sus pulsos emitidos y ajustar la tasa de emisión de pulsos $r\in[0,1]$, dependiendo de la proximidad de su objetivo.
- Aunque la intensidad del sonido puede variar de muchas maneras, asumimos que la intensidad del sonido varía desde un valor grande (positivo) $A_0$ hasta un valor mínimo constante $A_{min}$.

### Movimiento
- Las soluciones (posiciones) y las velocidades de los murciélagos en un instante $i$ son:
	- $x_i^t =x_i^{t-1}+v_i^t$
	- $v_i^t=v_i^{t-1}+(x_i^t-x_*)f_i$
	- $f_i=f_{min}+(f_{max}-f_{min})\beta$
- Donde $\beta$ es un [[vector]] aleatorio escogido de una distribución uniforme, $x_*$ es la mejor posición global que es seleccionada tras comprobar los $n$ murciélagos.
- Como el productor de $\lambda_if_i$ es el incremento de la velocidad, se puede usar tanto uno como otro para ajustar la velocidad mientras el otro es fijado, todo depende del problema de interés.
- En el documento original, se utiliza un rango para $f$ de $[0,100]$. A cada murciélago se le asigna una frecuencia uniformemente de $[f_{min},f_{max}]$.
- Una vez se ha seleccionado una mejor solución, se actualizan los vectores posición de cada murciélago:
	- $x_{new} = x_{old}+\epsilon A^t$
- Donde $\epsilon\in[-1,1]$ es un número aleatorio y $A^t$ es la intensidad sonora media de los murciélagos.

### Intensidad sonora y Emisión de pulso
- La intensidad sonora y el ratio de emisión de pulsos deben actualizarse según avance el número de iteraciones. $A_i$ debe decrementarse a la vez que el ratio $r_i$ se incrementa según el murciélago se acerca a su presa. Se actualizan del siguiente modo:
	- $\begin{equation}A_i^{t+1}=\alpha A_i^t,\quad r_i^{t+1}=r_i^0[1-exp(-\gamma t)]\end{equation}$
- Donde $\gamma, \alpha$ son constantes. 
- Estos valores solo se actualizan si las nuevas soluciones son mejores, de manera que se guíe hacia un óptimo.