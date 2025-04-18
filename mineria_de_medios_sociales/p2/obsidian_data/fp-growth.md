- #associationrules | #fpgrowth

# Concepto
- Crea una representación comprimida de la base de datos en una estructura con forma de árbol. Esta estructura consta de:
	- Tabla cabecera: Es una tabla de listas. Para cada *item* de la base de datos hay una lista que enlaza todos los nodos del árbol donde aparece.
	- Grafo de transacciones: Describe de forma abreviada todas las transacciones de la base de datos, indicando en cada nodo el soporte del *itemset* que se forma siguiendo el camino que va desde la raíz hasta dicho nodo.
- Una vez construido el árbol se utiliza la técnica divide y vencerás para extraer los *itemsets* frecuentes.

# Algoritmo
- Se calcula la frecuencia de cada *item*.
- Eliminamos los *items* que no superen el mínimo umbral.
- Ordenamos cada *itemset* de forma descendiente teniendo en cuenta la frecuencia de cada *item*.
- Construimos el árbol empezando por la raíz **null**:
	- Añadimos un nodo por cada *item* y a cada nodo se le añade un hijo, que es el siguiente *item* del *itemset* que se está recorriendo. Cada nodo debe tener la frecuencia del *item*. En la primera pasada será uno.
	- En el siguiente *itemset* hacemos lo mismo, incrementando la frecuencia de aquellos nodos que se han repetido **por el mismo camino**. Se crean nuevos nodos hijos de *items* que ya existían si el *item* padre es distinto. Por ejemplo:
![[fp-growth-example.png]]
- De esta forma se crea un árbol donde cada posible rama indica los distintos *itemsets* que hay.
- Una vez se ha creado el FP-Tree, se procede a la extracción de *itemsets* frecuentes.