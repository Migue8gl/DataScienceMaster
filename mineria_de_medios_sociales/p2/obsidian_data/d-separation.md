- #bayesiannetwork 

# Concepto
- La **D-separación** (dependencia-separación) es un concepto clave en redes bayesianas que permite identificar si dos conjuntos de nodos (variables aleatorias) son **independientes condicionalmente** dado un tercer conjunto de nodos. 
- Proporciona una forma estructural (basada en la topología de la red) para determinar qué información es relevante para predecir una variable, sin necesidad de calcular todas las probabilidades.

# Intuición
- Dos conjuntos de nodos, $X$ y $Y$, están **D-separados** por un conjunto de nodos $Z$ si, una vez conocido $Z$, $X$ y $Y$ son condicionalmente independientes. Esto significa que no podemos obtener más información sobre $Y$ a partir de $X$ si ya conocemos $Z$.
![[d-separation.png|500]]

# Algoritmo
- **Pregunto:** ¿Son $A$ y $B$ condicionalmente independientes dado un conjunto de evidencias $Z$?.
	- Sí, si $A$ y $B$ están separadas por $Z$.
	- Se consideran todos los caminos indirectos entre $A$ y $B$.
	- No hay ningún camino **activo**.
- Un camino es activo si cada **tripleta** que lo compone es activa.
	- Cadena causal $A\rightarrow B\rightarrow C$ donde $B$ no es observado.
	- Causa común $A\leftarrow B\rightarrow C$ donde $B$ no es observado.
	- Efecto común (*v-structure*) $A\rightarrow B\leftarrow C$ donde $B$ o uno de sus nodos descendentes son observados.
![[d-separation-algorithm-example.png|500]]
## Explicación
1. **Cadena causal**  
   $$ A \rightarrow B \rightarrow C: \text{Si $B$ es observado, el camino se bloquea porque conocer $B$ rompe la dependencia entre $A$ y $C$.} $$

2. **Causa común**  
   $$ A \leftarrow B \rightarrow C: \text{Si $B$ es observado, el camino se bloquea porque conocer $B$ explica $A$ y $C$ independientemente.} $$

3. **Estructura en V**  
   $$ A \rightarrow B \leftarrow C: \text{Si $B$ o un descendiente es observado, el camino se activa porque la observación induce dependencia entre $A$ y $C$.} $$
