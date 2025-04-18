- #firstorderlanguages | #lenguajesdeprimerorden | #lmd 

# Sintaxis del lenguaje 3.1
## Introducción 3.1.1
- Supongamos que establecemos que *todos los hombres son mortales* y que *Sócrates es un hombre*. Por tanto *Sócrates es mortal*. Este tipo de razonamientos se conocen como [[syllogism|silogismo]] y fue formulado por primera vez por Aristóteles.
- En este nuevo lenguaje las proposiciones atómicas ya no son objetos indivisibles, sino que vienen dadas por un sujeto y un predicado.
- Los objetos que puedan ser sujetos de alguna fórmula atómica se denominarán **términos**.
- Tendremos una serie de términos sobre los que podemos hacer afirmaciones (predicados).

## Descripción de un lenguaje de primer orden 3.1.2
### Definición 1
- Un lenguaje de primer orden $\mathscr{L}$ es una 4-tupla de conjuntos $(C,V,F,R)$.
	- A los elementos de $C$ se les llama símbolos de constante.
	- A los elementos de $V$ se les llama símbolos de variables.
	- A los elementos de $F$ se les llama símbolos de función.
	- A los elementos de $R$ se les llama símbolos de relación o predicado.
- Cada elemento de $F$ y $R$ está asociado con un número natural $x\in\mathbb{N}$.
- Además también tiene el conjunto de conectivas $\{\lor,\land,\implies,\iff,\neg\}$ y el conjunto de cuantificadores $\{\forall,\exists\}$.
- Para los símbolos de variable utilizaremos las últimas letras del alfabeto $(x,y,z,x_1,x_2,etc.)$.
- Para los símbolos de constante utilizaremos las primeras letras del alfabeto $(a,b,c,d,a_1,a_2,etc.)$.
- Los superíndices de función y relación son la aridad.
- Para los símbolos de función utilizaremos $(f¹,h²,g²,f,g,etc.)$ donde el superíndice puede ser cualquier $x\in\mathbb{N}$.
- Para los símbolos de predicado usaremos letras mayúsculas $(A,B,C,etc.)$.

### Definición 2
- Sea $\mathscr{L}=(C,V,F,R)$ un lenguaje de primer orden. Definimos el conjunto $T$ como sigue:
	- $C\subseteq T$.
	- $V\subseteq T$.
	- Si $f$ es un símbolo de función n-ario y $t_1,t_2,...,t_n\in T$ entonces $f(t_1,t_2,...,t_n)\in T$.
	- Todo objeto construido de otra forma **no** pertenece a $T$.
- A los elementos de $T$ se les conoce como términos del lenguaje.
- Sea el lenguaje:

![[first-order-language-example.png]]

![[first-order-terms-example.png]]

### Definición 3
- Sea $\mathscr{L}=(C,V,F,R)$ un lenguaje de primer orden cuyo conjunto de términos es $T$. Definimos el conjunto de fórmulas atómicas del lenguaje $\mathscr{L}$ como el conjunto $P(t_1,t_2,...,t_n)$. Siendo $P$ un símbolo de predicaro n-ario.
- Según el lenguaje puesto de ejemplo en la definición 2:

![[atomic-formulas-example.png]]

### Definición 4
- Sea $\mathscr{L}$ un lenguaje de primer orden. Definimos las fórmulas del lenguaje como sigue:
	- Toda fórmula atómica es una fórmula.
	- Si $\alpha$ y $\beta$ son fórmulas, también lo son $\alpha\lor\beta,\alpha\land\beta,\alpha\implies\beta,\alpha\iff\beta,\neg\alpha$.
	- Si $\alpha$ es una fórmula y $x$ es un símbolo de variable, entonces son fórmulas $\forall x\alpha,\exists x\alpha$.
	- Todo lo que no se haya construido siguiendo las reglas anteriores no es una fórmula.
- Los cuantificadores tienen prioridad sobre el resto de conectivas.

![[first-order-formulas-schema-example.png]]

- Podemos desmenuzar las fórmulas en subfórmulas. Algo visual para ello sería escribirlas en forma de árbol:

![[formula-example.png]]

- Sus subfórmulas serían:

![[formula-subformulas-tree-form-example.png]]

### Definición 5
- Supongamos que tenemos una subfórmula de la forma $Cx\beta$ donde $C$ es un cuantificador (universal o existencial). Se define el **radio de acción, dominio o ámbito** de este cuantificador como la subfórmula $\beta$.

### Definición 6
- Sea $\alpha$ una fórmula de un lenguaje de primer orden, y supongamos que tenemos una aparición de una variable $x$ en dicha fórmula. Diremos que dicha aparición es ligada si está en el radio de acción de un cuantificador $\forall x$ o $\exists x$. En caso contrario diremos que dicha aparición es libre.

### Definición 7
- Sea $\alpha$ una fórmula de un lenguaje de primer orden. Decimos que $\alpha$ es una sentencia si en $\alpha$ no hay ninguna ocurrencia libre de ninguna variable (o lo que es lo mismo, todas las ocurrencias son ligadas).**
- Una fórmula sin variables es una sentencia.

![[sentence-tree-form-example.png]]

# Semántica de un lenguaje de primer orden 3.2
## Estructuras 3.2.1
### Definición 8
- Sea $\mathscr{L}=(C,V,F,R)$ un lenguaje de primer orden. Una estructura $\varepsilon$ consiste en:
	- Un conjunto $D$ distinto del vacío, denominado dominio o universo.
	- Para cada símbolo de constante $a\in C$, un elemento $a^\varepsilon\in D$ (tenemos una aplicación $C\rightarrow D$).
	- Para cada símbolo de función n-ario $f$, una aplicación $f^\varepsilon:D^n\rightarrow D$.
	- Para cada símbolo de predicado n-ario $P$, una aplicación $P^\varepsilon:D^n\rightarrow\mathbb{Z}_2$.
- Dar una estructura es elegir los elementos sobre los que queremos que las fórmulas digan algo (matrices, personas, polinomios, etc.) y una vez hecho esto, asignar un significado a los símbolos de función, constante y relación.

![[first-orden-structure-example.png]]

### Definición 9
- Sea $\mathscr{L}$ un lenguaje de primer orden, y $\varepsilon$ una estructura. Una valoración es una aplicación $v:V\rightarrow D$.

### Definición 10
- Sea $\mathscr{L}$ un lenguaje de primer orden, sea $\varepsilon$ una estructura y $v$ una valoración. Entonces $v$ se extiende a una aplicación (que seguiremos llamando $v$):
- $v:T\rightarrow D$
	- Si $a\in C$ entonces $v(a)=a$ (para ser más precisos $v(a)=a^\varepsilon$).
	- Si $x\in V$ entonces $v(x)$ ya está definido.
	- Si $f$ es un símbolo de función n-ario y $t_1,t_2,...,t_n$ son términos para los que tenemos definido $v$ entonces se define $v(f(t_1,t_2,...,t_n))$ como $f(v(t_1),v(t_2),...,v(t_n))$.
- El elemento $v(t)$ diremos que es el valor del término $t$.
- Ejemplo: $x=0$ o $x\mapsto 0$ sería una valoración/asignación de $x$.

### Definición 11
- Sea $\mathscr{L}$ un lenguaje de primer orden. Una interpretación $I$ es un par $(\varepsilon,v)$ donde $\varepsilon$ es una estructura y $v$ una asignación/valoración.
- Una interpretación determina un valor de verdad para cada fórmula del lenguaje. Si $I(\varepsilon,v)$ es una interpretación y $\alpha$ una fórmula, denotaremos por $I_{\varepsilon}^v(\alpha)$ al valor de verdad de la fórmula.

### Definición 12
- Sea $\mathscr{L}$ un lenguaje primer orden, $I=(\varepsilon,v)$ una interpretación y $\alpha=P(t_1,t_2,...,t_n)$ una fórmula atómica. Definimos el valor de verdad de $\alpha$ bajo la interpretación $I$ como $I(\alpha)=P(v(t_1),v(t_2),...,v(t_n))$.

- Una vez definido el valor de verdad de una fórmula atómica, vamos a extenderlo a cualquier fórmula. Para eso, necesitamos antes ver cómo modificar el valor de una valoración en una variable. Sea $\mathscr{L}$ un lenguaje de primer orden, $\varepsilon$ una estructura y $v:V\rightarrow D$ una valoración. Para cada variable $x$ y cada elemento $e\in D$ definimos una nueva valoración $v_{x|e}:V\rightarrow D$ como:

	- $v_{x|e}(y)=\begin{cases}v(y)&y\neq x\\e&y=x\end{cases}$

- Es decir,  $v_{x|e}$ actúa igual que v sobre todas las variables salvo eventualmente $x$. A esto se le conoce como **variante**.
- Con esto podemos definir el valor de verdad de cualquier fórmula.

### Definición 13
- Sea $\mathscr{L}$ un lenguaje de primer orden. Sea $I=(\varepsilon,v)$ una interpretación, con $D$ el dominio.
- Sean $\alpha$ y $\beta$ dos fórmulas de $\mathscr{L}$ entonces los valores para las interpretaciones se calculan como [[propositional logic#Definición 4|en la lógica proposicional]]. Además:
	- $I^v(\forall x\alpha)=\begin{cases}1&\text{si para cualquier elemento }e\in D\text{ se tiene que }I^{v_{x|e}}(\alpha)=1\\0&\text{en otro caso}\end{cases}$
	- $I^v(\exists x\alpha)=\begin{cases}1&\text{si para algún elemento }e\in D\text{ se tiene que }I^{v_{x|e}}(\alpha)=1\\0&\text{en otro caso}\end{cases}$

### Lema 3.2.1
- **Lema de coincidencia**: Sea $\alpha$ una fórmula de un lenguaje de primer orden, y sea $\varepsilon$ una estructura. Supongamos que el conjunto de variables que tienen alguna ocurrencia libre en $\alpha$ es $\{x_1,x_2,...,x_m\}$ . Sean $v_1,v_2$ dos valoraciones tales que:
	- $v_1(x_i)=v_2(x_i)$ para $i=1,2,...,m$.
- Diremos que $I^{v_1}(\alpha)=I^{v_2}(\alpha)$.
- Es decir, a la hora de interpretar una fórmula, no importa como esta actúe sobre las variables ligadas, solo importa sobre las libres. Como consecuencia si la fórmula es una [[first order languages#Definición 7|sentencia]], su interpretación solo dependerá de la estructura, pues no tiene variables libres.

## Clasificación semántica de las fórmulas 3.2.2
### Definición 14
- Dada una fórmula $\alpha$ en un lenguaje de primer orden y una interpretación $I^v=(\varepsilon,v)$, se dice que es un *modelo* para $\alpha$ si $I^v(\alpha)=1$.

### Definición 15
- Sea $\alpha$ una fórmula de un lenguaje de primer orden, y sea $\varepsilon$ una estructura.
	- $\alpha$ es válida en $\varepsilon$ si para cualquier valoración $v$ se tiene que $I^v(\alpha)=1$.
	- $\alpha$ es satisfacible en $\varepsilon$ si para alguna valoración $v$ se tiene que $I^v(\alpha)=1$.
	- $\alpha$ es refutable en $\varepsilon$ si para alguna valoración $v$ se tiene que $I^v(\alpha)=0$.
	- $\alpha$ es no válida en $\varepsilon$ si para toda valoración $v$ se tiene que $I^v(\alpha)=0$.
- Si $\alpha$ es una [[first order languages#Definición 7|sentencia]], de forma que su valor de verdad no depende de la valoración, y $\varepsilon$ es una estructura, entonces, si $\varepsilon$ es satisfacible en la estructura, entonces también será válida en ella. Por el contrario, si es refutable, entonces es no válida en esa estructura.

### Definición 16
- Sea $\alpha$ una fórmula de un lenguaje de primer orden.
	- Se dice que $\alpha$ es universalmente válida si para cualquier estructura $\varepsilon$, esta fórmula es válida en $\varepsilon$. Es decir, para cualquier interpretación $\alpha$ es válida.
	- Se dice que $\alpha$ es satisfacible si existe alguna interpretación para la que $\alpha$ es válida (alguna estructura para la que $\alpha$ es satisfacible).
	- Definiciones equivalentes para refutable y contradicción.

### Ejemplo
- Demostrar que $\exists xP(x)\implies P(a)$ es satisfacible y refutable.
	- Primero debemos ver que la fórmula es una [[first order languages#Definición 7|sentencia]] (no tiene ninguna variable libre ya que todas las variables, es decir, $x$, están bajo el radio de acción del cuantificador existencial). Por ello para comprobar su satisfacibilidad tenemos que encontrar una estructura para la que la valoración de la fórmula sea satisfacible, es decir, que para alguna valoración se tenga que $I^v(\alpha)=1$. En este caso que se una [[first order languages#Definición 7|sentencia]] no nos aporta nada con respecto a lo que se nos pide.
	- $\varepsilon = \{D=\mathbb{N}, a=1, P(x)\equiv x \text{ es impar}\}$.
	- Buscamos una valoración en la que la fórmula sea cierta con esa estructura. Para $v(x)=1$ tenemos que $P(x)=1, P(a)=1, P(x)\implies P(a)$ y $\exists xP(x)=1, \exists xP(x)\implies P(a)=1$, por lo que ya sabemos que la fórmula es satisfacible. Es verdadera para cualquier $x$.
	- Para demostrar que es refutable simplemente buscamos una estructura en la que la fórmula sea falsa en alguna valoración (será falsa en todas las valoraciones pues la fórmula es una sentencia).
	- $\varepsilon=\{D=\mathbb{N}, a=1, P(x)=2·a \text{ es impar}\}$.
	- Sabemos que cualquier número multiplicado por 2, en este caso la constante $a$, es par, por lo que para cualquier valoración de $x$ tenemos que la fórmula es falsa. Por ello, la fórmula es refutable. Por ejemplo, $v(x)=1$ hace que $P(x)=0$ (en realidad cualquier $x$), por ello $\exists xP(x)=0$ y eso hace finalmente que $\exists xP(x)\implies P(a)=0$, ya que $P(a)=1$, el cálculo sería el siguiente -> $1+0+1·1=0$ (estamos en $\mathbb{Z}_2$).

### Ejemplo
- Demostrar que $\forall xP(x)\implies P(a)$ es universalmente válida.
	- La fórmula es una [[first order languages#Definición 7|sentencia]], por lo que cualquier valoración en una estructura $\varepsilon$ hará que esta sea o válida o contradicción. Si es universalmente válida, entonces en todas las estructuras $\varepsilon$ debe ser válida.
	- En un dominio cualquier $D$, con una constante $a$ cualquiera del dominio $D$, si para todo  $x$ se cumple $P(x)$ entonces se cumple también $P(a)$, esto es siempre cierto porque si $P(x)$ es siempre cierto para cualquier $v:V\implies D$, entonces al ser $a\in D$ está incluido en el conjunto de valoraciones, por lo que es cierto también, lo que nos da $1+1+1·1=1$. En cambio, si para cualquier valoración $P(x)$ no se cumple, tampoco se cumplirá $P(a)$, por lo que el resultado sería $1+0·0·0=1$. Es universalmente válida.

### Definición 17
- Sea $\Gamma=\{\gamma_1,\gamma_2,···,\gamma_n\}$ un conjunto de fórmulas. Se dice que $\Gamma$ es un conjunto satisfacible si existe una interpretación $I=(\varepsilon, v)$ tal que $I^v(\gamma_1) = I^v(\gamma_2) = ...= I^v(\gamma_n)=1$, es decir, que existe un [[first order languages#Definición 14|modelo]] para todas las fórmulas de $\Gamma$. Un conjunto no satisfacible es insatisfacible.

# Implicación semántica 3.3
- Misma problema que en la lógica proposicional, que a fin de cuentas es un subconjunto de la lógica de primer orden -> [[propositional logic#Implicación semántica 2.4.2|Implicación Semántica]].
- Pero ahora, para ver si $\gamma_1,\gamma_2,...,\gamma_3\models\alpha$ no podremos hacer como en la lógica propreposicional y tomar todas las interpretaciones en las que las premisas sean ciertas para ver qud ocurre con con la conclusión, pues el conjunto de las interpretaciones puede ser infinito.
- Lo que sí es cierto es que si encontramos una interpratación que haga ciertas todas las hipótesis, pero no la conclusión, entonces podemos concluir que $\Gamma\models\alpha$ no es cierto.

### Silogismos
- [[syllogism|Silogismo]] | #silogismo 

### Teorema 3.3.1
- Sea un conjunto de fórmulas $\Gamma$ y otra fórmula $\alpha$. Son equivalentes:
	- $\Gamma\models\alpha$
	- $\Gamma\cup\{\neg\alpha\}$ es insatisfacible.

### Teorema 3.3.2
- (Teorema de la deducción)
- Sea $\Gamma$ un conjunto de fórmulas (que podría ser vacío) de un lenguaje de primer orden, y $\alpha, \beta$, otras dos fórmulas. Entonces las siguientes afirmaciones son equivalentes:
	- $\Gamma\models\alpha\implies\beta$
	- $\Gamma\cup\{\alpha\}\models\beta$
- Al igual que en la lógica preposicional podemos resolverlo por el método de #david-putman o por resolución, pero si recordamos bien, en la lógica preposicional transformabamos las fórmulas en otras con determinada forma (clausulas). Vamos a hacer algo parecido, pero adaptado a la lógica de primer orden.

# Equivalencia lógica 3.4
### Definición 18
- Sean $\alpha,\beta$ dos fórmulas de un lenguaje de primer orden. Se dice que son lógicamente equivalentes si para cualquier interpretación $I=(\varepsilon,v)$ se tiene que $I(\alpha)=I(\beta)$.
- Si $\alpha$ y $\beta$ son lógicamente equivalente entonces escribiremos $\alpha\equiv\beta$.
- Si en una fórmula sustiuimos una subfórmula por otra lógicamente equivalente, entonces la fórmula resultante es lógicamente equivalente a la original.
- [[propositional logic#Equivalencia lógica 2.3|Equivalencias lógicas]].

# Algunas equivalencias lógicas 3.5
- Vamos a estudiar como se comportan los cuantificadores $\forall,\exists$ con respecto a los conectores lógicos $\lor,\land,\neg,\implies$.
- Dada una fórmula $\alpha$, un símbolo de variable $x$ y un término $t$, denotaremos como $a_{x|t}$ a la fórmula resultante de sustituir todas las ocurrencias libres de $x$ por $t$.

### Ejemplo
- Sea $\alpha=\forall x(\exists yR(x,f(y))\implies P(y,x))$.
	- $\alpha_{y|z}=\forall x(\exists yR(x,f(y))\implies P(z,x))$. Solo se ha sustituido en la última $y$, ya que las otras apariciones de esta variable son ligadas
	- $\alpha_{y|f(z)}=\forall x(\exists yR(x,f(y))\implies P(f(z),x))$.

## Negación y cuantificadores 3.5.1
- Vamos a justificar que para cualquier fórmula, las fórmulas $\neg\forall x\alpha$ y $\exists x\neg\alpha$ son equivalentes.
	- Consideremos el enunciado, *todos los números primos son impares*. Vamos a expresarlo en un lenguaje de primer orden, consideramos la estructura cuyo dominio son los números naturales y los significados de los símbolos de predicado son los siguientes: $Pr(x)\equiv x$ es primo e $Im(x)\equiv x$ es impar. Podemos traducir pues el enunciado anterior tal que así: $\forall x(Pr(x)\implies Im(x))$.
	- Sabemos que el enunciado es falso, lo que nos dice que en esta estructura el valor de verdad de $\neg\forall x(Pr(x)\implies Im(x))$ es uno.
	- El motivo por lo que la anterior fórmula es falsa es porque existe un número que es primo y que es par, este número es el **2**. Esto podría expresarse como $\exists x(Pr(x)\land\neg Im(x))$.
	- Lo que hace falsa a $\forall x(Pr(x)\implies Im(x))$ y verdadera a $\neg\forall x(Pr(x)\implies Im(x))$ es lo que hace verdadera a $\exists x(Pr(x)\land\neg Im(x))$. Si llamamos $\alpha$ a $Pr(x)\implies Im(x)$ entonces:
		- $\exists x(Pr(x)\land\neg Im(x))\equiv\exists x\neg(\neg Pr(x)\lor Im(x))\equiv\exists x\neg(Pr(x)\implies Im(x))\equiv\exists x\neg\alpha$
	- Por ello llegamos a la conclusión de que $\neg\forall x\alpha\equiv\exists x\neg\alpha$, es decir, significan lo mismo.
	- De otra forma tenemos que $\neg\exists x\alpha\equiv\forall x\neg\alpha$.

## Inclusión de $\lor$ o $\land$ en el radio de acción del cuantificador (I) 3.5.2
- Sean dos fórmulas $\alpha,\beta$ y consideramos la fórmula $\forall x\alpha\land\beta$. En esta última fórmula $\beta$ queda fuera del radio de acción del cuantificador universal. Vamos a ver formas de incluir esta fórmula dentro del radio de acción de $\forall x$.

| En $\beta$ no hay ocurrencias libres de $x$ | En $\alpha$ no hay ocurrencias libres de $x$ |
|:-----------------------------------:|:-----------------------------------:|
| $\forall x\alpha\land\beta\equiv\forall x(\alpha\land\beta)$ | $\alpha\land\forall x\beta\equiv\forall x(\alpha\land\beta)$ |
| $\forall x\alpha\lor\beta\equiv\forall x(\alpha\lor\beta)$ | $\alpha\lor\forall x\beta\equiv\forall x(\alpha\lor\beta)$ |
| $\exists x\alpha\land\beta\equiv\exists x(\alpha\land\beta)$ | $\alpha\land\exists x\beta\equiv\exists x(\alpha\land\beta)$ |
| $\exists x\alpha\lor\beta\equiv\exists x(\alpha\lor\beta)$ | $\alpha\lor\exists x\beta\equiv\exists x(\alpha\lor\beta)$ |

- El caso en el que la variable $x$ tenga alguna ocurrencia libre se tratará más adelante.

## Eliminación de cuantificadores 3.5.3
- Los cuantificadores sirven para cuantificar variables. Si la fórmula que tenemos no depende de la variable que el cuantificador mide, entonces podemos suprimir el cuantificador. Dicho de otra forma, si $\alpha$ es una fórmula que no tiene ocurrencias libres de la variable $x$ entonces $\forall x\alpha$ y $\alpha$ son equivalentes.
- Si una variable $x$ aparece en el radio de acción de dos cuantificadores $C_1x$ y $C_2x$ $(C_1,C_2\in\{\forall,\exists\})$, el que está más a la izquierda de ella no actúa sobre la variable.

## Cambio de variable 3.5.4
- Sabemos que al interpretar una fórmula con una variable ligada, la variable en cuestión no es significativa. Es lo mismo decir *existe un número x tal que x es par y x es primo* que decir *existe un número y tal que y es par y y es primo* $\rightarrow$ $\exists x(P(x)\land Pr(x))=\exists y(P(y)\land Pr(y))$.
- Queremos ver si una variable ligada puede ser cambiada por otra.
- Supongamos que $\alpha$ es una fórmula en la que **no hay ninguna ocurrencia de y (ni libre ni ligada)**. En ese caso: 
	- $\forall x\alpha\equiv\forall y\alpha_{x|y}$, (igual para cualquier cuantificador).

### Ejemplo:
- $\alpha=\forall x(P(x)\implies\exists xQ(x,a))$. Vamos a cambiar la variable $y$ del primer cuantificador $\rightarrow  \forall y(P(y)\implies\exists xQ(x,a))$. Como dijimos anteriormente, la segunda aparición de $x$ está cuantificada por $\exists x$, no se ve afectada por el primer cuantificador.
- Se recomienda no hacer cambios de variable con variables que aparezcan en la fórmula, así evitamos posibles errores al generar fórmulas no equivalentes.

## Agrupar cuantificadores 3.5.5
- Ahora nos encontramos con fórmulas de la forma $\forall x\alpha\land\forall x\beta$ y $\forall x\alpha\lor\forall x\beta$. Nos preguntamos si es posible la equivalencia de $\forall x(\alpha\lor\beta)$ con la segunda fórmula vista, por ejemplo. Vemos que no lo es. Ejemplos en lenguaje natural:
	- *Todo número entero es par o impar*.
	- *Todo número entero es par o todo número entero es impar*.
- Obviamente no son iguales.
- El cuantificador $\land$ para el ejemplo $\forall x\alpha\land\forall x\beta\equiv\forall x(\alpha\land\beta)$ si es verdad.
- Basándonos en esa equivalencia, en las vistas en la [[first order languages#Negación y cuantificadores 3.5.1|sección 3.5.1]] y en las leyes de De Morgan podemos obtener otra equivalencia:
	- $\exists x\alpha\lor\exists x\beta\equiv\exists x(\alpha\lor\beta)$.

## Inclusión de $\lor$ y $\land$ en el radio de acción de un cuantificador (II) 3.5.6
- En la anterior sección se dijo que se estudiaría el caso en el hubiese ocurrencias libres de la variable cuantificada en la fórmula. 
- Supongamos que tenemos una fórmula de la forma $\forall x\alpha\land\beta$ y que ahora la variable $x$ es libre en $\beta$. Entonces, elegimos una variable que no tenga ninguna ocurrencia (ni libre ni ligada) en $\alpha$ ni ninguna ocurrencia libre en $\beta$. Sea esta variable $y$. Sabemos según la [[first order languages#Cambio de variable 3.5.4|sección 3.5.4]] que $\forall x\alpha\equiv\forall y\alpha_{x|y}$. Entonces tenemos:
	- $\forall x\alpha\land\beta\equiv\forall y\alpha_{x|y}\land\beta\equiv\forall y(\alpha_{x|y}\land\beta)$.
- De esta forma ya hemos incluido la conectiva $\land$ y la fórmula $\beta$ en el radio de acción del cuantificador.

### Ejemplo
- $\alpha=\forall xP(x)\lor\exists yQ(x,y)$:
	- $\forall xP(x)\lor\exists yQ(x,y)\equiv$
	- $\forall zP(z)\lor\exists yQ(x,y)\equiv$
	- $\forall z(P(z)\lor\exists yQ(x,y))$.
- Como la variable $y$ no aparece en $P(z)$ entonces esta fórmula es equivalente a:
	- $\forall z\exists y(P(z)\lor Q(x,y))$.
- Podríamos haberlo hecho de otras formas también.

## Intercambio de cuantificadores 3.5.7
- Si $\alpha$ es una fórmula, entonces las fórmulas $\forall x\forall y\alpha\equiv\forall y\forall x\alpha$. Lo mismo podemos decir de $\exists x$. Estas equivalencias solo ocurren con mismo tipo de cuantificadores.

## Resumen de equivalencias 3.5.8

![[first-order-logic-equivalent-formulas.png]]

# Formas normales 3.6
- Vamos a obtener una fórmula clausulada, llamada así por su analogía con la forma clausulada de la lógica preposicional. Para llegar a ella primero debemos obtener la forma prenexa y la forma de Skolem.

## Formal normal prenexa 3.6.1
### Definición 19
- Sea $\alpha$ una fórmula. Se dice que $\alpha$ está en forma normal #prenexa si $\alpha$ es de la forma:
	- $C_1x_1C_2x_2...C_nx_n\beta$
- Donde $C_i$ es un cuantificador y $\beta$ es una fórmula sin cuantificadores. Es decir, una fórmula está en forma normal prenexa cuando todos los cuanficadores están al principio y su radio de acción cubre la fórmula hasta el final.
- Una fórmula sin cuantificadores está en forma normal prenexa.

### Ejemplo
- Dada la siguiente fórmula, transformarla a su forma normal prenexa. Sea la fórmula $\alpha=\forall xS(x)\implies\exists z\forall yR(z,y)$. 
	- $\forall xS(x)\implies\exists z\forall yR(z,y)\equiv$
	- $\neg\forall xS(x)\lor\exists z\forall yR(z,y)\equiv$ ([[first order languages#Equivalencia lógica 3.4|equivalencia lógica]])
	- $\exists x\neg S(x)\lor\exists z\forall yR(z,y)\equiv$ ([[first order languages#Negación y cuantificadores 3.5.1|equivalencia cuantificadores]])
	- $\exists x(\neg S(x)\lor\exists z\forall yR(z,y))\equiv$
	- $\exists x\exists z\forall y(\neg S(x)\lor R(z,y))$.
- Dada la siguiente fórmula, transformarla a su forma normal prenexa. Sea la fórmula $\beta=\forall x(R(x,y)\land\neg\forall yR(x,y))$.
	- $\forall x(R(x,y)\land\neg\forall yR(x,y))\equiv$
	- $\forall x(R(x,y)\land\exists y\neg R(x,y))\equiv$ ([[first order languages#Negación y cuantificadores 3.5.1|equivalencia cuantificadores]])
	- $\forall x(R(x,y)\land\exists z\neg R(x,z))\equiv$ (cambio de variable $y$ -> $z$)
	- $\forall x\exists z(R(x,y)\land\neg R(x,z))$.

### Teorema 3.6.1
- Sea $\alpha$ una fórmula de un lenguaje de primer orden. Entonces existe una fórmula equivalente a $\alpha$ y que está en forma prenexa.

## Forma normal de Skolem 3.6.2
### Definición 20
- Sea $\alpha$ una fórmula en un lenguaje de primer orden. Se dice que $\alpha$ está en forma normal de #skolem si está en forma normal prenexa y en ella no aparecen cuantificadores existenciales:
	- $\alpha=\forall x_1,\forall x_2,...,\forall x_n\beta$
- Donde $\beta$ es una fórmula sin cuantificadores.
- Dada una fórmula, no siempre es posible transformarla en otra fórmula Skolem equivalente, pero para el objetivo que se persigue (satisfabilidad) no es necesario.
- Para conseguir eliminar los cuantificadores existenciales se sustituye la variable cuantificada existencialmente por un término al mismo tiempo que se elimina el cuantificador existencial. El término por el que se sustituirá será:
	- **Caso 1**: Símbolo de constante si el cuantificador existencial que la acompaña no va precedido por ningún cuantificador universal. El símbolo elegido no debe aparecer previamente en la fórmula. Ej:
	- $\exists x\forall y(\neg S(x)\lor R(x,y)) \rightarrow\forall y(\neg S(a)\lor R(a,y))$.
	- **Caso 2**: Símbolo de función cuya aridad sea igual al número de variables cuantificadas universalemente y que preceden a la variable a sustituir. La función debe aplicarse a todas estas variables y el símbolo elegido no puede aparecer previamente. Ej:
	- $\forall x\exists z(R(x,y)\land\neg R(x,z))\rightarrow\forall x(R(x,y)\land\neg R(x,f(x)))$.

### Teorema 3.6.2
- Sea $\Gamma$ un conjunto de fórmulas, y sea $\Gamma^*$ el conjunto que resulta de sustituir cada fórmula de $\Gamma$ por su forma de Skolem. Entonces $\Gamma$ es insatisfacible si, y solo si, $\Gamma^*$ es insatisfacible.

## Forma clausulada 3.6.3
- Un literal es una fórmula atómica o el negado de una fórmula atómica. Si $\lambda$ es un literal, denotaremos $\lambda^c$ como su negado.
- El cierre universal de una fórmula sin cuanficadores es la fórmula que resulta de cuantificar universalmente todas las variables que aparecen en la fórmula. Algunos ejemplos de cierres universales para fórmulas son:
	- $P(a) \rightarrow P(a) \text{ (ella misma)}$.
	- $\neg H(x,g(x,a),y)\rightarrow\forall x\forall y\neg H(x,g(x,a),y)$.
- Una cláusula es el cierre universal de una [[disjunction|disyunción]] de literales. En la definición de cláusula se incluye el caso de cierre universal de un literal.
- También se incluye el caso de una disyunción de cero literales. Esta cláusula se denomina cláusula vacía y la denotaremos como $\square$. La cláusula vacía es insatisfacible.
- Una fórmula se dice que esta en forma clausulada si está descrita como [[conjunction|conjunción]] de cláusulas. No se incluye el caso de conjunción de cero cláusulas. Ej: $\forall x\forall y(P(x)\lor\neg Q(g(f(x),b),y))\land\forall xH(a,x,f(x))$.
- No toda fórmula tiene forma clausulada, sólo si es [[first order languages#Definición 7|sentencia]], ya que en la forma clausulada todas las ocurrencias de las variables son ligadas.
- Para calcular la forma clausulada de una sentencia procedemos como sigue:
	- Calculamos la forma normal de Skolem.
	- Sobre la fórmula que resulta de quitar los cuantificadores aplicamos algunas equivalencias hasta dejar la fórmula como una conjunción de disyunción de literales:
		- $\forall x_1\forall x_2...\forall x_n(\delta_1\land\delta_2\land...\land\delta_m)$
		- Donde $\delta_i$ es una disyunción de literales.
	- Utilizamos la [[first order languages#Resumen de equivalencias 3.5.8|equivalencia nº 16]] para distribuir los cuantificadores universales sobre las fórmulas.
	- De cada una de las subfórmulas $\forall x_1\forall x_2...\forall x_n\delta_i$ eliminamos los cuantificadores correspondientes a variables que no aparecen en $\delta_i$.

### Ejemplo
- Transformar la siguiente fórmula en su forma clausulada:
	- $\forall x\forall y(R(x)\lor(Q(x,y)\land P(y)\land R(x)))$.
- Podemos observar que es una [[first order languages#Definición 7|sentencia]], de forma que podemos calcular su forma clausulada.
- Vemos que la fórmula ya está en forma prenexa, de hecho, ya está en forma de Skolem al no existir cuantificadores existenciales en ella. Calculamos directamente su forma clausulada.
- $\forall x\forall y(R(x)\lor(Q(x,y)\land P(y)\land R(x)))\equiv$
- $\forall x\forall y((R(x)\lor Q(x,y))\land(R(x)\lor P(y))\land(R(x)\lor R(x)))\equiv$
- $\forall x\forall y((R(x)\lor Q(x,y))\land(R(x)\lor P(y))\land R(x))\equiv$
- $\forall x\forall y(R(x)\lor Q(x,y))\land\forall x\forall y(R(x)\lor P(y))\land\forall x\forall yR(x)\equiv$
- $\forall x\forall y(R(x)\lor Q(x,y))\land\forall x\forall y(R(x)\lor P(y))\land\forall xR(x).$
- Con esto hemos seguido todos los pasos para transformar la fórmula inicial en una fórmula en forma clausulada.
- Cuando tenemos un conjunto de sentencias y calculamos la forma normal de Skolem y la forma clausulada, sabemos que todas las variables están cuantificadas universalmente, por lo que podemos omitir los cuantificadores.

### Teorema 3.6.3
- Sea $\Gamma=\{\gamma_1,\gamma_2,...,\gamma_n\}$ un conjunto de sentencias. Supongamos que para $\gamma_i\in\Gamma$ su forma clausulada es $C_{i1}\land C_{i2}\land...\land C_{im_i}$. Sea $\Gamma^{**}$ el conjunto formado por todas las cláusulas que aparecen en la forma clausulada de cada una de las fórmulas de $\Gamma$, es decir,
	- $\Gamma^{**}=\{C_{11},C_{12},...,C_{1m_1},C_{21},C_{22},...,C_{2m_2},...,C_{n1},C_{n2},...,C_{nm_n}\}$.
- Entonces $\Gamma$ es insatisfacible si, y solo si, $\Gamma^{**}$ es insatisfacible.

### Lema 3.6.1
- Lo normal es trabajar con sentencias, pero en caso de que algunas formas tengan variables libres, se procede siguiendo los siguientes lemas.
- Sea $\Gamma=\{\gamma_1,\gamma_2,...,\gamma_n\}$ un conjunto de fórmulas y $\gamma=\gamma_1\land\gamma_2\land...\land\gamma_n$. Entonces $\Gamma$ es satisfacible si, y solo si, $\gamma$ es satisfacible. De otra manera, $\Gamma$ es insatisfacible si, y solo si, $\gamma$ es contradicción. 

### Lema 3.6.2
- Sea $\alpha$ una fórmula en un lenguaje de primer orden, y sea $x$ una variable que aparece libre en la fórmula $\alpha$. Entonces $\alpha$ es contradicción si, y solo si, $\exists x\alpha$ es una contradicción.

### Teorema 3.6.4
- Sea $\Gamma=\{\gamma_1,\gamma_2,...,\gamma_n\}$. Supongamos que el conjunto de variables que tienen alguna ocurrencia libre en las fórmulas de $\Gamma$ es $\{x_1,x_2,...,x_m\}$. Entonces $\Gamma$ es insatisfacible si, y solo si, la fórmula $\exists x_1\exists x_2...\exists x_n(\gamma_1\land\gamma_2\land...\land\gamma_n)$ es una contradicción.
- La fórmula $\exists x_1\exists x_2...\exists x_n(\gamma_1\land\gamma_2\land...\land\gamma_n)$ es una sentencia, y por tanto a esta fórmula le podemos hacer su forma clausulada y reducir el problema a estudiar si un conjunto de cláusulas es satisfacible o insatisfacible.
- Sin embargo trabajar con $\exists x_1\exists x_2...\exists x_n(\gamma_1\land\gamma_2\land...\land\gamma_n)$ es bastante engorroso, por lo que podemos sustituir las variables libres $x_i$ que nos encontremos por símbolos de constante (no puede estar ya en alguna fórmula del conjunto $\Gamma$) y debemos sustituir todas las apariciones de $x_i$ con el mismo símbolo. De esta forma ya son sentencias.