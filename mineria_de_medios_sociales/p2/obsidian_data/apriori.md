- #associationrules | #frequentintemset

# Principio
- El método de cálculo de *itemsets* frecuentes se basa en que si un *itemset* es frecuente, lo serán todos sus subconjuntos.
- Este principio se mantiene gracias a la propiedad **anti-monótona** de la medida de soporte:
	- $\forall X,Y:(X\subseteq Y)\rightarrow s(X)\ge s(Y)$
- Gracias a esto, sabemos que si un *itemset* es infrecuente, entonces todos sus superconjuntos inmediatos serán infrecuentes. Con ello, se puede podar al grafo de generación de *itemsets*. 
![[apriori-prune.png|550]]

# Algortimo
- $L_k$ -> Conjunto de $k$-*itemsets* que son frecuentes (largos).
- $C_k$ -> Conjutno de $k$-*itemsets* que pueden ser frecuentes (candidatos).
- Supongamos $k=1$:
	- Generar $L_1$ (*itemsets* frecuentes de logitud $1$).
- Repetir hasta que no se identifiquen *itemsets* nuevos:
	- Generear el conjunto $C(k+1)$ de *itemsets* candidatos a partir del conjunto $L_k$ de *itemsets* frecuentes (combinando los *itemsets* frecuentes que solo se diferencien en el último *item*).
	- Calcular soporte de cada candidato recorriendo la BD.
	- Eliminar los candidatos que son infrecuentes e incluirlos en el conjunto $L_{k+1}$.
	- $k=k+1$.
![[apriori-example.png|570]]
![[apriori-example-manual.png|500]]

# Desventajas
- Elección del umbral del mínimo soporte: Umbrales muy bajos darán lugar a muchos *itemsets* frecuentes y y de mayor longitud. Esto lleva a una necesidad de mayor espacio y de tiempo de ejecución.
- Número de *items* en la base de datos: Más espacio para almacenar el soporte de cada *itemset*.
- **Apriori** hace muchas pasadas por la base de datos, por lo que el tiempo de ejecución puede incrementar con el número de transacciones.