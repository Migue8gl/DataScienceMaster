- #dl | #attention | #transformers
- **Links**:
	- [Explicación de atención](https://arxiv.org/abs/1409.0473)

# Concepto
- La idea detrás de este concepto, es la de considerar para cada [[embedding#^b65400|token]] la totalidad de tokens de la entrada (solamente los tokens con mayor probabilidad tendrán peso en la actualización del token actual) para cambiar su significado semántico en función del contexto.
- Básicamente es un mecanismo en el que una secuencia de caracteres es capaz de darse contexto a sí misma por medio de la relación de las palabras entre sí.
## Atención original
- Se introdujo el concepto de atención mediante el uso de un **Encoder-Decoder RNN**.
- En el Encoder se lee una secuencia de entrada, representada como una secuencia de [[vector|vectores]] $X=(x_1,...,x_{T_x})$. Esta secuencia de vectores es transformada en un vector $c$.
- Sea una [[recurrent neural networks|RNN]] tal que:
	- $h_t=f(x_t,h_{t-1})$
	- $c=q(\{h_1,...,h_{T_x}\})$
	- Donde $h_t$ es un [[hidden state|estado oculto]] en el tiempo $t$ y $c$ es un vector formado por la secuencia de todos los estados ocultos. $q$ y $f$ son funciones no lineales.
- El Decoder es entrenado para predecir la siguiente palabra $y_{t^\prime}$ dado un vector de contexto $c$ y todas las palabras predichas anteriormente $\{y_1, ..., y_{t^\prime-1}\}$. 
	- $p(y)=\prod_{t=1}^Tp(y|\{y_1,...,y_{t-1}\},c)$
	- Donde se considera la [[joint probabilites|probabilidad conjunta]] de una serie de [[conditional probability|probabilidad condicionales]] para el token $y_t$ dado los tokens anteriores y un contexto.
- Como resumen, en el paper se trata de pasar de la secuencia $X$ a la secuencia $Y$, pues es una traducción.
- En una RNN la probabilidad $p(y_t|y_1,...,y_{t-1},c)=g(y_{t-1},s_t,c)$ donde $g$ es una función no lineal (normalmente multicapas) y $s_t$ es un [[hidden state|estado oculto]] en tiempo $t$.
- En el paper original no hay un solo vector de contexto $c$, sino que hay uno por cada token objetivo -> $g(y_{t-1},s_i,c_i)$. Cada estado oculto se calcula como:
	- $s_i=f(s_{i-1}, y_{i-1}, c_i)$
- El vector de contexto $i$ se calcula como una serie de anotaciones $(h_1, ..., h_{T_x})$ en las que el Encoder mapea la secuencia de entrada $X$.
- Cada vector $h_i$ contiene información de la secuencia completa con un focalización especial en las palabras que procedentes y subsecuentes a la palabra/token $i$.
- Cada vector de contexto se construye como una suma ponderada de cada una de esas anotaciones/estados ocultos:
	- $c_i=\sum_{j=1}^{T_x}a_{ij}h_j$
	- Los pesos $a_{ij}$ se calculan como:
		- $a_{ij}=\frac{e^{e_{ij}}}{\sum_{k=1}^{T_x}e^{e_{ik}}}$ [[softmax]]
		- Donde $e_{ij}=a(s_{i-1}, h_j)$. Es un modelo de [[alignment|alineamiento]] que puntua como de bien se alinean los inputs de la posición $j$ y el output de la posición $i$.
		- El modelo $a$ es una Red Neuronal que se entrena con el resto del sistema.
![[attention-mechanism-ilustrated.png]]
- En este imagen puede diferenciarse el Encoder (abajo), el mecanismo de atención per se (computación del vector de contexto) y el Decoder (arriba).
- $h$ es estado oculto del Encoder, $s$ del Decoder.