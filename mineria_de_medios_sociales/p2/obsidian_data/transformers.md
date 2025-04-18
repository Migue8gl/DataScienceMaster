- #transformers | #dl | #nlp
- **Links**:
	- [¿Qué es un transformer?](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=4s)
	- [Atención](https://www.youtube.com/watch?v=wjZofJX0v4M&t=1180s)
	- [Attention is all you need paper](https://arxiv.org/abs/1706.03762)

## Instalación librerías
- Pytorch:
	- Instalación de [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
	- `python -m pip install torch`
- Transformers:
	- `python -m pip install transformers`

## Conceptos clave
![[transformer.png|400]]
### Word embedding
- [[embedding]]

### Capa de atención
- Lo que da potencia a los **transformers** es la capa de [[attention|atención]]. 
- La matriz de palabras embebidas es capaz de almacenar en vectores cada uno de los tokens, dandole así un significado semántico según el vector de representación en el espacio. 
- Pero una palabra puede significar muchas cosas según el contexto del resto de palabras alrededor suya, es por ello que la capa de atención se vale de un [[window size|tamaño de contexto]] en el que cada token dentro de la capa cambia la información de los vectores de la capa.

### Dot product
- [[dot product]]
- Puede entenderse el producto escalar de dos vectores producidos por dos tokens, como una forma de similitud entre tokens (similitud semántica).
- Un ejemplo intuitivo:
	- $\vec{v}(cats)-\vec{v}(cats) = \vec{plural}$
	- $\vec{plural}\cdot\vec{x}$ -> Nos daría un escalar que mide "como de plural" es una palabra. Obviamente es un ejemplo que en la práctica no se corresponde totalmente, pero la intuición es esa.

## Matemática detrás de las cabezas de atención
- Cada bloque de atención está formado por múltiples cabezas o unidades que utilizan el producto escalar, debido a la información de similitud que ofrece esta operación sobre las representaciones vectoriales de los token.
- Para cada unidad de atención, el modelo aprende tres matrices de peso: las matrices de **Query** ($W_Q$), **Key** ($W_K$) y **Value** ($W_V$).  
- Para cada token $i$, la representación del token de entrada $x_i$ se multiplica con cada una de las tres matrices de peso para producir un vector de consulta $q_i = x_i W_Q$, un vector de clave $k_i = x_i W_K$ y un vector de valor $v_i = x_i W_V$. 
	- Puede pensarse que la matriz **query** o consulta contiene información de qué debe buscarse, mientras que la matriz **key** o clave contiene información de lo que se busca. De esta forma, si hay similitud ([[dot product]]) entre $q_i\cdot k_j$ puede interpretarse como que la consulta $i$ es respondida por la clave $j$ y por tanto, el token $j$ tiene información sobre el contexto de $i$. En la práctica ambas matrices son aprendidas en el entrenamiento y no es posible interpretar toda la información que contienen.

![[attention-head-mechanism.png]]

- Los pesos de atención se calculan utilizando los vectores de consulta y clave: el peso de atención $a_{ij}$ de token $i$ a token $j$ es el producto escalar entre $q_i$ y $k_j$.   
- En el paper original no se tiene en cuenta para el contexto que tokens anteriores influencien a tokens posteriores, es decir, todos los tokens que cuyo índice $j<i$. Para ello se aplica un proceso llamado **masking**. A cada celda de atención por dot-product cuyo índice de token sea $i<i$ se le da el valor de $-\infty$. Cuando se aplique **softmax** después, quedará reducido a una probabilidad de $0$.
- Estos pesos de atención se normalizan dividiéndolos por la raíz cuadrada de la dimensión de los vectores de clave, $\sqrt{d_k}$, lo que [[numerical stability|estabiliza]] los gradientes durante el entrenamiento, y luego se pasan por una función **softmax**.  
- La salida de la unidad de atención para el token $i$ es la suma ponderada de los vectores de valor de todos los tokens, ponderada por $a_{ij}$, la atención desde el token $i$ a cada token.

 ![[attention-head-giving-new-context.png]]

- Cuando se multiple la matriz de valores $W_v$ por el token $x_j$ puede entenderse como: *Si este token es relevante para la actualización del contexto de $x_i$, ¿qué es necesario para trasladar el nuevo significado al token $x_i$?*
- Se actualiza el token $i$ cuyo vector es $x_i$ de la siguiente manera:
	- $x_i^\prime=x_i+\Delta x_i$
	- Donde $\Delta x_i$ es la ya mencionada suma ponderada de vectores de valor.
- El cálculo de la atención para todos los tokens se puede expresar como una operación de matriz utilizando la función [[softmax]], lo cual es útil para el entrenamiento debido a las optimizaciones  computacionales. Las matrices $Q$, $K$ y $V$ se definen como las matrices donde las filas $i$ son vectores $q_i$, $k_i$ y $v_i$ respectivamente. Luego podemos representar la atención como:  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$