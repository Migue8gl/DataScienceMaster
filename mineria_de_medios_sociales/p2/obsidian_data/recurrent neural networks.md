- #neuralnetworks | #recurrent

# Concepto
- Las redes neuronales son redes usadas para procesar datos secuenciales usando la técnica de recurrencia. 
- La **unidad recurrente** es un tipo de unidad que mantiene un "[[hidden state]]", que es esencialmente, una forma de memoria. Esta se actualiza mediante los valores del *input* actual y el de los estados ocultos anteriores. Este *feedback* hace que se pueda aprender de *inputs* pasados.

# Problema del gradiente que se desvanece
- La redes, sobre todo de este tipo, sufren un problema conocido como el problema del gradiente que se desvanece. Esto ocurre cuando, al realizar la retropropagación, los pesos se van empequeñeciendo hasta quedar casi reducidos.

# Tipos
- Existen multitud de tipos de redes recurrentes, entre ellas:
	- **Bidireccionales:** Permiten al modelo procesar un *token* en el contexto de lo que venía antes de este y después de este. 
	- **Encoder-decoder**: Como por ejemplo *seq2seq*. El *encoder* procesa una secuencia de entrada y la transforma en estados ocultos y el *decoder* procesa los estados ocultos a una secuencia final (con ayuda de un buen operador de [[attention|atención]]). Es la base de los [[transformers]].