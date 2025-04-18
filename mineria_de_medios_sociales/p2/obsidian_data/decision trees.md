- #decisiontrees | #ml

# Introducción
- Los árboles de decisión, ya sean para regresión o clasificación, son un tipo de algoritmo de aprendizaje automático que representa la toma de decisiones como un árbol hasta llegar a la clase o etiqueta correspondiente al *input* dado (en el caso de clasificación).
- Existen dos tipos de nodos en el árbol:
	- Los nodos **decisión**, que representan una condición sobre las variables. Es una forma de dividir los datos.
	- Los nodos **hoja**, que representan la etiqueta o categoría final.
- La intuición básica detrás del algoritmo es la de intentar obtener los nodos de decisión más puros posibles (maximizar), esto es, los que sean más discriminadores entre clases o los que aporten más información.
![[decision trees-purity-of-nodes-example.png]]
- Para obtener un **score** sobre cada nodo, se realizan cálculos basados en la [teoría de la información](https://en.wikipedia.org/wiki/Information_theory).
- La forma de cuantizarlo es con la entropía:
	- $Entropy=\sum_i^N-p_ilog(p_i)$
- Donde $p_i$ es la probabilidad de la clase $i$. 
- El valor más alto de entropía es $1$, que significa la mayor impureza, mientras que $0$ sería la mayor pureza.
- Para calcular que *split* es mejor, se calcula la **ganancia de información**:
	- $IG=E(parent)-\sum_i^Nw_iE(child_i)$
- Donde $E$ es la entropía y $w$ es el peso que corresponde al tamaño relativo de cada nodo hijo. En el ejemplo hay $20$ puntos, por tanto, el tamaño relativo del primer nodo de la izquierda es de $\frac{14}{20}$, ya que solo tiene $14$ puntos totales.
![[decision trees-information-gain-example.png]]
- Otro criterio para medir la impureza es el de [[gini impurity|Gini]].
- Los árboles de decisión son un tipo de algoritmo que tiene [[variance and standard deviation|varianza]] muy alta, por lo que cualquier cambio en los datos o parámetros pueden dar resultado a modelos muy distintos.