- #associationrules | #nonsupervised | #datamining

# Problemas semánticos
## Problemas derivados de los datos
- Si los datos no son apropiados las reglas pueden ser dudosas o inútiles. Esto puede ocurrir por varios motivos.
- **Falta de variabilidad**: Hay *items* muy poco o demasiado frecuentes.
- **Representatividad**: Los datos deben comprender todos los casos que se pretenden estudiar con un número suficiente de casos.
- **Sesgos muestrales** -> [[bias|sampling bias]].
- **Factores ocultos**: Como la estacionalidad, los *items* no considerados, etc.
- **Valores perdidos** -> [[missing values]].
## Problemas derivados de los usuarios
- No disponibilidad de expertos en los datos para la valoración de las reglas.
- **Confusión de semánticas**:
	- Confusión con **dependencias simétricas** (por influencia estadística).
	- Diversos tipos de **implicación lógica**. Las reglas de asociación implican tendencias y no implicaciones.
	- **Causalidad**: Tendencia no implica causalidad.
## Problemas derivados de las medidas
- Los *items* con soporte alto dan lugar a reglas con soportes altos en el consecuente -> reglas inútiles. Si sabemos que $A$ tiene un soporte de $0.89$ y este da lugar a una regla $A\rightarrow B$ con soporte $0.86$, en realidad, esta regla no aporta nada, pues el *item* $A$ aparece un $89\%$ de las veces en el conjunto de datos mientras que $A\land B$ ha reducido su porcentaje de soporte. A priori parece alto, pero no tiene por qué ser bueno.
- La confianza, al estar basada en frecuencias, no detecta cuando el soporte del consecuente es muy alto.
![[advanced association rules-bad-support.png|550]]

# Métricas avanzadas
- Es deseable que las medidas de interés tengan las siguientes propiedades:
	- $lift(A\rightarrow B)=1$ cuando son independientes ($supp(A\rightarrow B)=supp(A)\cdot supp(B)$).
	- $lift(A\rightarrow B)$ crece monotonamente con el $supp(A\rightarrow B)$ cuando se mantiene el resto de valores.
	- $lift(A\rightarrow B)$ decrece monotonamente con $supp(A)$ o $supp(B)$ cuando se mantiene el resto de valores.
## Confianza confirmada
- Se define para $A\rightarrow B$ como $Conf(A\rightarrow B) - Conf(A\rightarrow \lnot B)$
- ¿Hasta qué punto es útil $A$ para predecir la presencia de $B$?
- Su rango es de $[-1,1]$ donde $0$ significa imposible predecir (independencia), $1$ significa que $A$ predice $B$  y $-1$ que $A$ predice $\lnot B$.
## Convicción
- Se define para $A\rightarrow B$ como $\frac{supp(A)\cdot supp(\lnot B)}{supp(A\rightarrow\lnot B)}$
- Valores de $1$ significan independencia estadística. Valores negativos dependencia negativa.
## Factor de certeza
- $FC(A \to C) =\frac{Conf(A\to C)-supp(C)}{1-supp(C)}\hspace{3mm}\text{si}\hspace{3mm}Conf(A\to C)\ge supp(C)$
- $FC(A \to C) =\frac{Conf(A\to C)-supp(C)}{supp(C)}\hspace{3mm}\text{si}\hspace{3mm}Conf(A\to C)<supp(C)$
- Proviene del ámbito de los sistemas expertos.
- Es una medida de implicación. Mide la variación de nuestra creencia en $C$ cuando se cumple $A$ con respecto a la creencia a priori en $C$. Valor $0$ implica independencia estadística y su valor va de $[-1,1]$.
## Yule's Q
- $\LARGE{\frac{supp(AC) \cdot supp(\neg A \neg C) - supp(A \neg C) \cdot supp(\neg AC)}{supp(AC) \cdot supp(\neg A \neg C) + supp(A \neg C) \cdot supp(\neg AC)}}$
- Esta medida representa la relación entre dos eventos dicotómicos (sí o no) relacionados positivamente.
- Su rango es $[-1, 1]$. Valor $0$ significa independencia estadística; valores negativos dependecia negativa; y valores positivos dependencia positiva. 
- Cumple la mayoría de las propiedades de la literatura para las medidas de interés.
## Diferencia absoluta de confianza
- $Conf(A\to C) - supp(C)$
- Es una medida de implicación, va de $[-1,1]$ y cero significa independencia estadística.
## Ratio de confianza
- $1-\frac{Conf(A\to C)}{supp(C)}$
- Es una medida de implicación, va de $[-1,1]$ y cero significa independencia estadística.
- Especialmente adecuada para descubrir reglas que corresponden a itemsets poco frecuentes.
## Diferencia de información
$$\left(-sop(C) \log_2 sop(C) - sop(\neg C) \log_2 sop(\neg C)\right) - \left(-conf(A \Rightarrow C) \log_2 conf(A \Rightarrow C) - conf(A \Rightarrow \neg C) \log_2 conf(A \Rightarrow \neg C)\right)$$
- Cada parte es una medida de información basada en entropía. La primera considera la información dada solo por $C$, la segunda la dada por $C$ en presencia de $A$. Se mide la ganancia (o pérdida) de información sobre $C$ al conocer $A$.
- Es una medida de implicación. 
- Se ve afectada por el soporte.

# Medidas subjetivas
- Las **medidas objetivas** consideran únicamente los datos disponibles para evaluar las reglas.  
- Por el contrario, las **medidas subjetivas de cumplimiento** valoran el interés de las reglas no solo en términos de cumplimiento, sino también con base en otras características deseables, como que el conocimiento sea no trivial, novedoso y potencialmente útil.  
- Estas medidas deben tener en cuenta el conocimiento previo del usuario, incluyendo sus creencias y necesidades.
- Aquellas reglas que contradigan las creencias del usuario son reglas inesperadas y, como tales, interesantes.
- Hay que especificar cómo representar las creencias del usuario y cómo medir lo inesperado.
- Habitualmente mediante estadísticos y distancias entre los esperados por el usuario y los reales.
## Reglas inesperadas
- ### Medidas de la distancia sintáctica
- Este enfoque se centra en la **distancia** o diferencia entre las nuevas reglas y el conocimiento previo o conjunto de creencias del usuario. La idea es medir cuán similar o diferente es una regla nueva en comparación con lo que el usuario ya espera o conoce.
- Por ejemplo, si los consecuentes de una regla son los mismos que los esperados por el usuario, pero los antecedentes son muy distintos, entonces esta regla se consideraría interesante.
### Contradicción lógica: 
- Usa una medida objetiva para indicar lo que el usuario espera, y después se analiza si hay alguna diferencia con los grados esperados por el usuario de dichas medidas.
![[advanced association rules-logic-contradiction.png]]
# Interpretaciones
- Esta sección se corresponde con el marco formal de las reglas de asociación, es decir, la definición teórica de las reglas de forma abstracta. Para ello hay que asociar dicha abstracción a los datos y crear una asociación entre los datos y las reglas, de forma que generen una interpretación.
- Por ejemplo, los datos suelen darse de forma tabular como $(salario, alto)\rightarrow(estudios, superiores)$. Es la forma más normal.
- También pueden representarse las negaciones de *items*, de forma que representan la ausencia del *item* en el *itemset*.
- Otra forma son las **reglas jerárquicas**, donde existen grupos de *items* que pertenecen a *items* de clase superior, representando esta relación en forma de jerarquía. Dependiendo de la situación es posible que sea más conveniente recurrir a un nivel superior que a uno inferior o viceversa.
![[advanced association rules-hierarchical-interpretation.png|500]]
- Las **reglas secuenciales** se usan cuando existe un orden prefijado en los *items* de las transacciones. Por ejemplo, si $A, B,C$ aparecen en este orden específico, entonces $X$. Útil para datos secuenciales, como textos.
- Las **reglas cuantitativas** se utilizan para datos estructurados con dominios numéricos. Su principal problema es el soporte bajo y el valor semántico. Es útil representar el dominio como intervalos y generar pares $(atributo, intervalo)$. 
- Existen muchas más interpretaciones.

# Reglas de asociación difusas
- [[fuzzy association rules]]