- #dl | #autoencoder | #ml

# Concepto
- Los **autoencoders** son un tipo de arquitectura de red neuronal comúnmente usada en *deep learning* en tareas de compresión de datos, detección de anomalías, eliminación de ruido y detección facial.
- Son sistemas *self-supervised* cuyo objetivo de entrenamiento (codificación) es el de reducir la dimensionalidad de los datos y después, de forma precisa, reconstruirlos (decodificar) usando esa información comprimida.
- El objetivo principal de la red es el de extraer las [[latent space|variables latentes]] y descartar el ruido.

# Arquitectura básica
## Encoder
El codificador extrae variables latentes de los datos de entrada _x_ y las convierte en un [[vector]] que representa el espacio latente $z$. En un autoencoder básico ("*vanilla*"), cada capa sucesiva del codificador tiene menos nodos que la anterior; al pasar por cada capa, los datos se comprimen, reduciendo sus dimensiones.
Variantes de los autoencoders utilizan términos de regularización, como funciones que imponen _sparsity_ penalizando el número de nodos activados en cada capa, para lograr la reducción de dimensionalidad.
## Bottleneck
El "cuello de botella" es la capa de salida del codificador y la capa de entrada del decodificador. Contiene el [[latent space|espacio latente]]: una representación comprimida y de menor dimensión de los datos de entrada. Este cuello de botella es esencial para evitar que el decodificador simplemente copie o memorice los datos de entrada, lo que evitaría que el **autoencoder** aprenda.
## Decoder
El decodificador usa esa representación latente para reconstruir los datos originales, invirtiendo el proceso del codificador: en una arquitectura típica, cada capa sucesiva tiene un número creciente de nodos activos.
![[autoencoder.png]]

# Tipos
- [[variational autoencoder]]