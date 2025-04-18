- #entropy | #impurity | #tree | #gini

## Concepto
- La métrica de impureza de *gini*  mide la frecuencia con la que un elemento elegido al azar de un conjunto se etiquetaría incorrectamente si se etiquetara de forma aleatoria e independiente según la distribución de etiquetas del conjunto. 
- Alcanza su mínimo (cero) cuando todos los casos del nodo caen en una única categoría objetivo. 

## Fórmula
- $I_G(p)=\sum_{i=1}^J(p_i\sum_{k\neq i}^J(1-p_i))=1-\sum_{i=1}^{J}p_{i}^2$
- Donde $p_i$ es la proporción de items clasificados con la clase $i$ y $J$ el número de clases.

![[gini impurity-example.png]]