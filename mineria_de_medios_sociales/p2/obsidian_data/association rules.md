- #associationrules | #nonsupervised | #datamining

# Conceptos
## Básicos
- **items**: Son parejas (atributo, valor).
- **transacciones**: Registros.
- **reglas**: *item1* -> *item2* (todo el que *item1* tiene *item2*). Estas reglas deben cumplir que dado $X\rightarrow Y$, entonces $X\cap Y=\varnothing$.
- **itemset**: conjunto de $1$ o más *items*.
![[association rules-table-to-list.png|500]]
## Medidas
### Soporte
- De un *itemset* $X$:
	- De un *itemset* $X$ es una medida de frecuencia.
	- El soporte del *itemset* $X$ es número de ocurrencias entre el total de transacciones de la BD, es decir, es la probabilidad de $X$ en la BD.
	- Un *itemset* es **frecuente** cuando el número de soporte es superior a un umbral definido.
	- $Soporte(X)=\frac{N_X}{N}$
- De una regla de asociación ($X\rightarrow Y$):
	- Frecuencia con la que ocurre el *itemset* $X\cup Y$.
	- Probabilidad del *itemset* $X\cup Y$ en el conjunto de transacciones $p(X\land Y)$.
	- $Soporte(X\rightarrow Y)=\frac{N_{X\land Y}}{N}$
- Tomas valores entre $[0,1]$.
### Confianza
- De una regla de asociación $X\rightarrow Y$:
	- Es una medida de cumplimiento.
	- Ratio de veces que los *items* en $Y$ aparecen en transacciones que contienen $X$.
	- $Confianza(X\rightarrow Y)=\frac{Soporte(X\rightarrow Y)}{Soporte(X)}$
	- $Confianza(X\rightarrow Y)=p(Y|X) = \frac{p(X\land Y)}{p(X)}$
- Ocurre en los rangos $[0,1]$. Si es uno, siempre que ocurre $X$ ocurre $Y$.
![[association rules-example-metrics.png]]
- Dado un conjunto de transaciones $T$, el objetivo del proceso de extracción es encontrar todas las reglas que tengan:
	- Soporte $\geq$ mínimo soporte (minSup)
	- Confianza $\geq$ mínima confianza (minConf)
- Los umbrales de minSup y minConf son dados por el experto del problema.
### Lift
- $\text{lift}(A \rightarrow B) = \frac{\text{confianza}(A \rightarrow B)}{\text{soporte}(B)}$
- El _lift_ es una métrica en las reglas de asociación que mide cuán relevante es una regla en comparación con lo que se esperaría por azar. En otras palabras, evalúa si la presencia de un antecedente $A$ realmente aumenta (o disminuye) la probabilidad de que el consecuente $B$ esté presente.
- **Lift = 1**: La ocurrencia de $B$ es independiente de la ocurrencia de $A$. Es decir, que $A$ y $B$ no están relacionados; saber que $A$ ocurre no da más (ni menos) información sobre $B$.
- **Lift > 1**: La presencia de $A$ aumenta la probabilidad de que $B$ ocurra. Esto indica una relación positiva y, por lo tanto, la regla puede ser interesante. Cuanto mayor sea el lift, más fuerte es la asociación entre $A$ y $B$.
- **Lift < 1**: La presencia de $A$ reduce la probabilidad de que $B$ ocurra, lo que indica una relación negativa. En este caso, saber que $A$ ocurre puede implicar que $B$ es menos probable.

# Métodos de cálculo
- Fuerza bruta: Se calculan todas las reglas posibles, se calculan todas las métricas para cada regla y se eliminan aquellas que no superen los umbrales. Esto es computacionalmente costoso y nunca válido.
![[association rules-brute-force.png|600]]
- [[apriori|Método apriori]].
- [[fp-growth|Método FP Growth]].

# Itemsets maximales y cerrados
- El número de conjuntos de *itemsets* frecuentes crece de forma exponencial y esto crea un problema de almacenamiento. Por ello se requieren representaciones alternativas que permitan reducir el conjunto inicial, pero que nos dejen generar todos los *itemsets* frecuentes a partir de ellas.
## Maximales
- Los *itemsets* maximales son aquellos *itemsets* frecuentes para los que ninguno de sus superconjuntos inmediatos son frecuentes. 
- A partir de estos *itemsets* se pueden extraer todos los frecuentes, ya que son las combinaciones de *items* que se puedan formar a partir de ellos.
- Lo malo es que hay que volver a calcular el soporte de los *itemsets* generados.

## Cerrados
- Son aquellos *itemsets* frecuentes para los que ninguno de sus superconjuntos inmediatos tienen un soporte igual al de ellos.
- A partir de los cerrados se pueden obtener todos los *itemsets* frecuentes y, además, cualquier subconjunto de ellos que no sea otro *itemset* cerrado tiene el mismo soporte que ellos, por lo que no se deberá calcular el soporte de estos *itemsets*.
- Lo malo es que son más numerosos que los maximales y se necesita más espacio para almacenarlo.

# Generación de reglas
- Una vez obtenidos todos los *itemsets* frecuentes, se pueden generan regla de asociación usando todas las posibles combinaciones entre la parte antecedente y la parte consecuente, es decir, si el *itemset* es de tamaño $k$, existen unas $2^k-2$ reglas posibles.
- De todas formas, se suelen usar más las reglas con un solo atributo en el consecuente, pues son más interpretables.
- La medida de confianza no posee la propiedad anti-monótona. Esto significa reglas subconjunto de otras pueden tener distintas confianzas (tanto mayores como menores). Ej: -> $Conf(ABC\rightarrow D)\ne Conf(AB\rightarrow D)$.
- Las reglas construidas a partir del mismo *itemset* si tienen la propiedad anti-monótona:
	- $L=\{A,B,C,D\}$
	- $conf(BCD\rightarrow A) \ge conf(CD\rightarrow AB) \ge conf(D\rightarrow ABC)$

# Aspectos avanzados
- [[advanced association rules]]