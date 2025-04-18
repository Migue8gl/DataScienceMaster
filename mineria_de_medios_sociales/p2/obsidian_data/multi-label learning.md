- #multi-labellearning

# Contexto
- Es aquel problema en el que para cada instancia existen múltiples etiquetas válidas. Cada instancia se asocia a varias etiquetas.
- Puede aplicarse a problemas de clasificación y problemas de regresión.

# Métodos
## Transformación del problema
- Adapta un problema multi etiqueta a un problema de una sola etiqueta, de forma que se requiere un pre-procesado, pero se usan algoritmos convencionales.
## Transformación del algoritmo
- Adaptar un algoritmo que produce *outputs* de una sola etiqueta para que produzca múltiples.

# Algoritmos
## Binary Relevance
- Se entrena un clasificador binaria para cada etiqueta. Las **desventajas** son que no modela la dependencia entre etiquetas y es un mal algoritmo para el problema de desbalance de datos, ya que algunas etiquetas son mucho más comunes que otras.
## Label Powerset
- Trata cada combinación única de etiquetas como una única etiqueta para un problema de una sola etiqueta. La principal **desventaja** es que no es muy escalable y hay poca representación de clases raras.
## Pairwise Binary
- Para cada par de etiquetas $(Y_{i}​,Y_{j})$, se entrena un clasificador binario que decide cuál de las dos es más probable en una instancia dada. La **desventaja** es que se necesitan $\frac{m(m-1)}{2}$ clasificadores y se necesitan estrategias de desempate.
## Copy Weight
- Se duplica cada instancia tantas veces como etiquetas tenga. Cada duplicado se asocia con una única etiqueta y se le asigna un peso. El peso se calcula como $\frac{1}{Y(i)}$​, donde $Y(i)$ es el número de etiquetas asociadas con la instancia original.

# Classifier Chains
- Este método implica construir una cadena de clasificadores, donde cada clasificador predice una etiqueta basándose no solo en las características originales ($X$), sino también en las predicciones de las etiquetas anteriores en la cadena.
## Ensemble of classifier chains
- Un *ensemble* de cadenas de clasificadores funciona mejor ya que no hace asunciones sobre el orden de las etiquetas a la hora de predecir a otras y además captura mejor las relaciones entre ellas.

# Label distribution learning
- Es una generalización de *multi-label* que aprende las distribuciones de las diferentes etiquetas para una instancia dada.