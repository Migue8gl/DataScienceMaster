- #propositionallogic | #logicapreposicional | #lmd 

# Lenguaje Proposicional 2.1
## Introducción 2.1.1
### Definición 1
- La lógica es la ciencia que estudia los razonamientos correstos. Por tanto se ocupa de:
	- Inferencias válidas o razonamientos correctos.
	- Concepto de definición.
	- Concepto de significado.
	- Concepto de verdad.

## Elementos del lenguaje proposicional 2.1.2
- Proposiciones básicas.
- Operadores lógicos o conectivas.
- Sintaxis del cálculo proposicional.

### Definifición 2
- Sea X un conjunto finito distinto del vacío. Definimos el conjunto $Form(X)$ como sigue:
	- Todo elemento de X pertenece a $Form(x)$ -> $x\subseteq Form(X)$.
	- Si $\alpha,\beta\in Form(X)$, entonces son fórmulas: $\alpha\lor\beta, \alpha\land\beta,\alpha\implies\beta,\alpha\iff\beta,\neg\alpha$.
	- Cualquier objeto que no se haya obtenido a partir de las reglas anteriores no forma parte de $Form(X)$.
- Los elementos de $Form(X)$ se denominan fórmulas proposicionales. Los elementos de X se denominan proposiciones atómicas.
- Normalmente usaremos letras del alfabeto latino para hablar de las proposiciones atómicas y letras del alfabeto griego para hablar de las fórmulas.

### Subfórmulas

![[subformulas-propositional-logic.png]]

# Semántica de la lógica proposicional 2.2
## Interpretaciones 2.2.1
### Definición 3
- Dado una lenguaje proposicional construido sobre el conjunto X, una interpretación es una aplicación $I:X\to\mathbb{Z}_2$.
- Una interpretación lo que hace es asignar un valor de verdad a una proposición atómica.
- $I(p)=0$ significaría que bajo la interpretación $I$ la proposición $p$ es falsa.

### Definición 4
- Dado un lenguaje proposicional cuyo conjunto de fórmulas atómicas es X, y dado una interpretación $I:X\to\mathbb{Z}_2$, extendemos la aplicación $I$ a $Form(X)$ como sigue:
	- $I(\alpha\lor\beta)=I(\alpha)+I(\beta)+I(\alpha)\cdot I(\beta)$
	- $I(\alpha\land\beta)=I(\alpha)\cdot I(\beta)$
	- $I(\alpha\implies\beta)=1+I(\alpha)+I(\alpha)\cdot I(\beta)$
	- $I(\alpha\iff\beta)=1+I(\alpha)+I(\beta)$
	- $I(\neg\alpha)=1+I(\alpha)$
- Al estar en $\mathbb{Z}_2$ las operaciones son algebráicas en ese espacio y no booleanas, es decir, operaciones como $1+1$ son $0$ y no $1$.
- Utilizando el $1$, la suma y el producto obtenemos el polinomio de #gegalkine.

## Tabla de verdad para una fórmula 2.2.2
- Podemos crear una tabla con los $2^n$ mundos posibles. La tabla para las cinco conectivas sería:

![[truth-table-propositional-logc.png]]

- Se recogen todos los valores de verdad de una fórmula en la que solo aparece una conectiva.

## Clasificación de fórmulas 2.2.3
### Definición 5
- Sea $\alpha$ una fórmula de un lenguaje proposicional:
	- $\alpha$ es una **tautología** si para cualquier interpretación se tiene que $I(\alpha)=1$
	- $\alpha$ es **satisfacible** si existe al menos una interpretación para la que $I(\alpha)=1$.
	- $\alpha$ es **refutable** si existe al menos una interpretación para la que $I(\alpha)=0$.
	- $\alpha$ es **contradicción** si para cualquier interpretación $I(\alpha)=0$.
	- $\alpha$ es **contingente** si es satisfacible y refutable.
- Si α es una fórmula, no significa lo mismo decir $\alpha$ no es tautología que decir $\neg\alpha$ es tautología. En el primer caso estamos diciendo que $\alpha$ es refutable, mientras que en el segundo que $\alpha$ es contradicción. Es decir, $\alpha$ es refutable si, y sólo si, $\alpha$ no es tautología; y $\alpha$ es contradicción si, y sólo si, $\neg\alpha$ es tautología.

# Equivalencia lógica 2.3
### Definición 6
- Sean $\alpha,\beta$ dos fórmulas de un lenguaje proposicional. Se dice que son lógicamente equivalentes si para cualquier interpretación $I$ se tiene que $I(\alpha)=I(\beta)$.
- Escribiremos pues $\alpha\equiv\beta$.

![[equivalent-formulas-propositional-logc.png]]

# Consecuencia lógica 2.4
## Conjuntos satisfacibles e insatisfacibles 2.4.1
- Posiblemente el concepto **más importante de la lógica**.

### Definición 7
- Sea $\Gamma=\{\gamma_1,\gamma_2,···,\gamma_n\}$ un conjunto de fórmulas de un lenguaje proposicional. Se dice que $\Gamma$ es **satisfacible** si existe una interpretación $I$ para la que todas las fórmulas de $\Gamma$ sean verdad.
- Si $\Gamma=\emptyset$ entonces $\Gamma$ es satisfacible.
- Si no es satisfacible es insatisfacible.

### Proposición 2.4.1
- Sea $\Gamma=\{\gamma_1,\gamma_2,···,\gamma_n\}$ un conjunto de fórmulas. Entonces son equivalentes:
	- $\Gamma$ es insatisfacible.
	- $\gamma_1\land\gamma_2\land···\land\gamma_n$ es contradicción.
	- Para cualquier $I$, $I(\gamma_1\land\gamma_2\land···\land\gamma_n)=0$.
	- Para cualquier interpretación $I$, $\prod_{i=1}^nI(\gamma_i)=0$.

## Implicación semántica 2.4.2
- Cuando las premisas o hipótesis son ciertas, la conclusión tiene que serlo.

### Definición 8
- Sea $\Gamma$ un conjunto de fórmulas de un lenguaje proposicional y sea $\alpha$ una fórmula del mismo lenguaje. Decimos que $\alpha$ es consecuencia lógica de $\Gamma$ si para cualquier interpretación para la que todas las proposiciones de $\Gamma$ sean ciertas se tiene que $\alpha$ es también verdadera.
- Se escribe $\Gamma\models\alpha$ y se lee $\alpha$ es consecuencia lógica de $\Gamma$.
- A las fórmulas de $\Gamma$ se les llama premisas o hipótesis y a $\alpha$ conclusión o tesis.
- Para comprobar que las premisas implican la tesis debemos mirar aquellas interpretaciones que sean verdaderas para todas las premisas.

### Teorema 2.4.1
- Sea $\Gamma$ un conjunto de fórmulas y $\alpha$ otra fórmula. Son equivalentes:
	- $\Gamma\models\alpha$
	- $\Gamma\cup\{\neg\alpha\}$ es insatisfacible.

## Teorema de la deducción 2.4.3
### Teorema 2.4.2
- Sea $\Gamma$ un conjunto de fórmulas (podría ser vacío) de un lenguaje proposicional, y $\alpha,\beta$, otras dos fórmulas. Entonces las siguientes afirmaciones son equivalentes:
	- $\Gamma\models\alpha\implies\beta$
	- $\Gamma\cup\{\alpha\}\models\beta$
- El teorema de la deducción nos dice cómo actuar en un problema de implicación semántica cuando la tesis es una implicación.

![[deduction-theorem-equivalent-propositional-logic.png]]

# Forma clausulada de una fórmula 2.5
- Vamos a dar dos métodos más con los que resolver [[propositional logic#Implicación semántica 2.4.2|el problema de la implicación semántica]]. Hemos visto ya dos, el de las [[propositional logic#Tabla de verdad para una fórmula 2.2.2|tablas de verdad]] y el que usa ecuaciones en $\mathbb{Z}_2$. Los siguientes algoritmos utilizan la #cláusula.

### Definición 9
- Un literal es una [[propositional logic#Definifición 2|proposición atómica]] o su negada.
- Si $\lambda$ es un literal, su negado es $\lambda^c$.
- Una cláusula es una [[disjunction|disyunción]] de literales, de forma que no haya literales que provengan de la misma proposición atómica. Si aparece el literal $\lambda$, este solo puede aparecer una vez y no puede aparecer $\lambda^c$. Todo literal es una cláusula.
- La disyunción de cero literales la llamamos cláusula vacía y se nota como $\square$.
- Una fórmula está en **forma clausulada** si está escrita como [[conjunction|conjunción]] de cláusulas.

### Teorema 2.5.1
- Sea $\alpha$ una fórmula que no es [[propositional logic#Clasificación de fórmulas 2.2.3|tautología]]. Entonces existe otra fórmula $\beta$, lógicamente equivalente a $\alpha$, y que está en forma clausulada.

### Algoritmo de la forma clausulada
- Si tenemos una fórmula de la forma $\alpha_1\iff\alpha_2$, la sustituimos por $(\alpha_1\implies\alpha_2)\land(\alpha_2\implies\alpha_1)$.
- Si tenemos una fórmula de la forma $\alpha_1\implies\alpha_2$, la sustituimos por $\neg\alpha_1\lor\alpha_2$.
- Cualquier fórmula de la forma $\neg\neg\alpha$, la sustituimos por $\alpha$.
- Introducimos la conectiva $\neg$ dentro de los paréntesis utilizando las [[propositional logic#Equivalencia lógica 2.3|equivalencias lógicas]]:
	- $\neg(\alpha_1\lor\alpha_2)\equiv\neg\alpha_1\land\alpha_2$
	- $\neg(\alpha_1\land\alpha_2)\equiv\neg\alpha_1\lor\alpha_2$
- Si tenemos una fórmula de la forma $\alpha_1\lor(\beta_1\land\beta_2)$, la sustituimos por $(\alpha_1\lor\beta_1)\land(\alpha_1\lor\beta_2)$.
- A partir de aquí aplicamos recursivamente sobre las subfórmulas que sigan teniendo $\lor$ fuera de los paréntesis y/o $\land$ dentro.
- Conjunciones deben quedar dentro y disyunciones fuera de los paréntesis.
- Eliminamos redundancias. Eliminamos literales por [[boole algebra#Propiedades|idempotencia y absorción]].

![[clause-form-example.png]]

- A continuación un ejemplo de como proceder según los pasos anteriormente explicados.

![[clause-form-algorithm.png]]

# El problema de la implicación semántica 2.6

- Hemos definido lo que era la [[propositional logic#Implicación semántica 2.4.2|implicación semántica]]. Vamos a desarrollar técnicas que nos permitan dar respuesta a este problema dentro de la lógica proposicional.
- Vamos a estudiar primero si un conjunto de fórmulas es satisfacible o no.

### Teorema 2.6.1
- Sea un conjunto $\Gamma$ de fórmulas, $\Gamma^`$ un conjunto de fórmulas que se obtiene sustituyendo cada fórmula de $\Gamma$ por una forma clausulada de esa fórmula, y $\Gamma^{``}$ el conjunto que resulta de sustituir cada fórmula de $\Gamma^`$ por las cláusulas que la forman.
- Si $\Gamma$ es insatisfacible los otros dos conjuntos también lo son.
- Estudiamos pues si un conjunto de cláusulas es satisfacible o no para determinar si un conjunto de fórmulas es satisfacible.
- Junto con el [[propositional logic#Teorema 2.4.1|teorema 2.4.1]], podemos transformar el problema de implicación semántica en un problema de satisfacibilidad de cláusulas.

### Algoritmo de Davis-Putnam 2.6.1
#### Nomenclatura
- $C$ es una cláusula, $\lambda$ es un literal que aparece en la cláusula y $C-\lambda$ es una cláusula en la que no aparece el literal $\lambda$. Si $C$ es una cláusula con un único literal se llama cláusula #unit. También existe la clásula vacía -> $\square$. Si $C$ es una cláusula de solo un literal, si hacemos $C-\lambda$ obtenemos la cláusula vacía.

#### Reglas
- Dado $\sum$ como un conjunto de cláusulas y $\lambda$ como un literal definimos un nuevo conjunto $\sum_\lambda$ . Este último conjunto lo subdividimos en los siguientes:
	- $\sum_{\lambda_1}$ -> conjunto formado por todas las cláusulas que tienen el literal $\lambda$.
	- $\sum_{\lambda_2}$ -> conjunto formado por todas las cláusulas que tienen el literal $\lambda^c$.
	- $\sum_{\lambda_3}$ -> conjunto formado por todas las cláusulas que no cumplan lo anterior. No tienen ni $\lambda$ ni $\lambda^c$.
- El conjunto $\sum_\lambda$ se forma de la siguiente forma:
	- Si $C\in\sum_{\lambda_1}$ entonces eliminamos esa cláusula.
	- Si $C\in\sum_{\lambda_2}$ entonces añadimos la cláusula $C-\lambda^c$ al conjunto $\sum_\lambda$, es decir, eliminamos $\lambda^c$ de la cláusula.
	- Si $C\in\sum_{\lambda_3}$ añadimos $C$ al conjunto.
- De forma que:
	- Sea $\sum$ un conjunto de cláusulas. Supongamos que en $\sum$ hay una cláusula unit $\lambda$. Entonces $\sum$ es insatisafacible si, y sólo si, $\sum_\lambda$ es insatisfacible.
	- Sea $\sum$ un conjunto de cláusulas. Supongamos que en $\sum$ hay un [[pure literal|literal puro]] $\lambda$. Entonces $\sum$ es insatisafacible si, y sólo si, $\sum_\lambda$ es insatisfacible.
	- Sea $\sum$ un conjunto de cláusulas. Sea $\lambda$. un literal que aparece en alguna cláusula de $\sum$. Entonces $\sum$ es insatisafacible si, y sólo si, $\sum_\lambda$ y $\sum_{\lambda^c}$ son insatisfacible.
- El algoritmo de #david-putman nos dice si un conjunto es insatisfacible o no, y en caso de serlo, nos da una interpretación $I$ para la que todas las cláusulas de $\sum$ son ciertas.
- Ejemplo:

![[davis-putman-example.png]]

- Si la cláusula vacía pertenece a todas las hojas, el conjunto $\sum$ es insatisfacible. Si hay alguna hoja que sea el conjunto vacío entonces el conjunto es satisfacible. Si es satisfacible y tomamos los literales desde la hoja hasta la raíz y los hacemos ciertos ya tenemos una interpretación para la que todas las cláusulas son ciertas.

### Método de resolución 2.6.2
- Es un método sintáctico, a partir de dos fórmulas con una determinada estructura vamos a obtener nuevas fórmulas que son consecuencia de éstas.

### Teorema 2.6.2
- Sea $\alpha$, $\beta$, $\gamma$ tres fórmulas en un lenguaje proposicional. Entonces:
	- $\{\alpha\lor\beta,\neg\alpha\lor\gamma\}\models\beta\lor\gamma$ 

### Definición 10
- Sean $C_1$ y $C_2$ dos cláusulas. Supongamos que $\lambda$ es un literal tal que aparece en la cláusula $C_1$ y $\lambda^c$ aparece en la cláusula $C_2$. Una cláusula que sea equivalente a $(C_1-\lambda)\lor(C_2-\lambda^c)$ es lo que se conoce como #resolvente de $C_1$ y $C_2$.
- El [[propositional logic#Teorema 2.6.2|teorema 2.6.2]] nos dice que si $C_1$ y $C_2$ son cláusulas y $R$ es una resolvente suya, entonces $\{C_1,C_2\}\models R$.
- Ejemplo:

![[resolvent-example.png]]

- El método de resolución lo que pretende es, dado un conjunto de cláusulas, obtener resolventes de dichas cláusulas que son añadidas al conjunto. Todas estas nuevas cláusulas son consecuencia lógica del conjunto de partida.

### Definición 11
- Sea $\Gamma$ un conjunto de cláusulas y $C$ una cláusula. Una deducción (por resolución) de $C$ a partir de $\Gamma$ es una sucesión de cláusulas $C_1,C_2,...,C_n$ donde:
	- $C_n$ = $C$.
	- $C_i\in\Gamma$ es una resolvente de dos cláusulas del conjunto $\{C_1,...,C_{i-1}\}$.

### Teorema 2.6.3
- (Principio de resolución). Sea $\Gamma$ un conjunto de cláusulas. Entonces $\Gamma$ es insatisfacible si, y sólo si, hay una deducción por resolución de la cláusula vacía.

### Estrategias de resolución
#### Estrategia por saturación
- No es una estrategia de por sí, trata de calcular resolventes hasta que no se puedan calcular más. Puesto que en  un conjunto de $n$ fórmulas podemos llegar a obtener $3^n$ cláusulas, este método puede hacerse muy largo.
- Por seguir un orden, calculamos desde el conjunto inicial $\sum_0$ todas las resolventes posibles. Las añadimos al conjunto, obteniendo un nuevo conjunto $\sum_1$. Después repetimos el proceso hasta llegar al conjunto $\sum_n$.
- Se omiten las tautologías.

#### Resolución lineal
- Dada una deducción de una fórmula por resolución diremos que es *lineal* si para cada nueva resolvente que obtenemos hacemos uso de la obtenida en la etapa inmediatamente anterior.

![[lineal-deduction-example.png]]

- Siempre que tengamos una deducción de la cláusula vacía (conjunto insatisfacible) podremos hacer una deducción lineal hasta ella.

#### Estrategia Input
- Dado un conjunto de cláusulas $\sum$ se dice que una deducción por resolución es *input* si al menos una de las dos cláusulas usadas para calcular la resolvente pertence a $\sum$.
- Esta estrategia nos acota el número de posibles resolventes, el problema es que esta estrategia no es completa, es decir, podemos estar ante un problema insatisfacible y no llegar a la cláusula vacía por medio de esta estrategia.
- Existe una situación en la que esta estrategia si es completa.

### Definición 12
- Dado un lenguaje proposicional contruido sobre el conjunto $X$:
	- Un literal se dice positivo si es una fórmula atómica ($\lambda\in X$).
	- Un literal se dice negativo si es el negado de una fórmula atómica.
	- Una cláusula se dice negativa si todos sus literales son negativos.
	- Una cláusula se dice cláusula de #horn si tiene exactamente un literal positivo.
	- Un conjunto de clásulas es de Horn si tiene exactamente una cláusula negativa y el resto de cláusulas son de Horn.

### Teorema 2.6.4
- Sea $\sum$ un conjunto de cláusulas de un lenguaje proposicional que es un conjunto de Horn. Entonces $\sum$ es insatisfacible si, y sólo si, existe una deducción lineal-input de la cláusula vacía que se inicia en la cláusula negativa. 
- Todo conjunto formado por cláusulas de Horn es satisfacible (que no conjunto de Horn).
- El teorema no afirma que todo conjunto de Horn sea insatisfacible, solo que si es así, se puede llegar a una deducción lineal-input de la cláusula vacía.
- Algunos conjuntos pueden transformarse en conjuntos de Horn, de forma que el primer conjunto es insatisfacible solo si el segundo lo es. Para ello escogemos una cláusula que, después de cambiar algunos literales por sus complementarios, se transformará en la cláusula negativa y el resto en cláusulas de Horn. Si esto no ocurre seguimos con otra cláusula.
