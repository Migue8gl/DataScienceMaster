- #unificactionandresolution | #unificacionyresolución | #lmd

# Unificación 4.1
- En esta sección nos centraremos en el estudio de, dados dos o más literales, estudiar si es posible transformarlos de forma que queden iguales. Haremos esto mediante *sustituciones*, en las que se intercambian variables por [[first order languages#Definición 2|términos]].

## Sustituciones 4.1.1
- Recordemos que un literal es una fórmula atómica o la negación de una fórmula atómica.

### Definición 1
- Sea $\alpha$ un literal, $x$ una variable con alguna ocurrencia en la fórmula $\alpha$ y $t$ un término cualquiera. La sustitución elemental de $x$ por $t$ es la transformación de la fórmula $\alpha$ en otra fórmula en la que cada ocurrencia de la variable $x$ es sustituida por el término $t$.
- Dicha sustitución la representaremos como $\sigma(x|t)$, y la fórmula resultante de aplicar la sustitución $\sigma$, como $\sigma(\alpha)$.
- Ejemplo: 
	- En la fórmula $R(f(x),a,g(h(x),y))$ realizamos la sustitución $(x|g(a,y))$. Nos queda pues -> $R(f(g(a,y)),a,g(h(g(a,y)),y))$.
- En principio, al sustituir una variable por un término, dicha variable puede aparecer en el término sustituto. Podemos además hacer composiciones de sustituciones -> $\sigma_0\sigma_1...\sigma_n$.
- El orden de composición es importante, no es lo mismo $\sigma_0\sigma_1$ que $\sigma_1\sigma_0$.

### Definición 2
- Sea $\alpha$ un literal en un lenguaje de primer orden. Una sustitución en $\alpha$ es la transformación que consiste en sustituir algunas (o todas) las variables que aparecen en $\alpha$ por términos del lenguaje.
- Si $x_1,x_2,...,x_n$ son variables que vamos a sustituir, y $t_1,t_2,...,t_n$ los términos por los que sustituimos, entonces representaremos la sustitución como:
	- $(x_1|t_1;x_2|t_2;...;x_n|t_n)$

## Unificadores 4.1.2
### Definición 3
- Sean $\alpha_1,\alpha_2,...,\alpha_n$ literales. Un unificador para tales fórmulas es una sustitución $\sigma$ de forma que $\sigma(\alpha_1)=\sigma(\alpha_2)=...=\sigma(\alpha_n)$.
- Un conjunto de literales se dice **unificable** si existe un unificador para ellos. Caso contrario, dicho conjunto no es unificable.
- Ejemplo:
	- Las fórmulas $P(x)$ y $P(a)$ son unificables por medio de $(x|a)$.
	- Las fórmulas $P(x)$ y $P(f(x))$ no son unificables, pues para cualquier unificador $(x|t)$ tendríamos que $\sigma(P(x))=P(t)$ y para $\sigma(P(f(x)))=P(f(t))$, por tanto nunca serán iguales.
	- Para las fórmulas $Q(x,a)$ y $Q(y,b)$ no hay unificador posible, pues las constantes no son sustituibles, de manera que ninguna transformación hará posible que ambas fórmulas sean iguales.

### Definición 4
- Sean $\alpha_1,\alpha_2,...,\alpha_n$ literales en un lenguaje de primer orden, y sea $\sigma$ una sustitución. Se dice que $\sigma$ es un **unificador principal** para $\alpha_1,\alpha_2,...,\alpha_n$ si:
	- $\sigma$ es un unificador para $\alpha_1,\alpha_2,...,\alpha_n$.
	- Si $\tau$ es otro unificador para $\alpha_1,\alpha_2,...,\alpha_n$, entonces existe una sustitución $\tau_1$ de forma que $\tau=\tau_1\sigma$
- Es decir, $\sigma$ es un unificador principal si todos los unificadores pueden obtenerse a partir de $\sigma$.

## Algoritmo de unificación
- Este algoritmo nos permite un cálculo automático de un unificador más general (principal) para un conjunto de literales.
- Para comenzar con el algoritmo, primero ha de definirse el *conjunto de discordancia*.

### Definición 5
- Dados dos literales construidos sobre el mismo símbolo de predicado, se define el conjunto de discordancia como el conjunto formado por los primeros términos (recorridos los literales de izquierda a derecha) en los que difieren. En el caso en que los literales sean iguales, el conjunto de discordancia es vacío.
- Para hacer el conjunto de discordancia de $\lambda_1,\lambda_2,...,\lambda_n$ con $n\geq 3$ se hace el conjunto de discordancia de $\lambda_1$ y $\lambda_2$. Si es vacío, el de $\lambda_1$ y $\lambda_3$, y así hasta que encontremos uno no vacío (si lo hubiera).
- En el caso de que todos los literales coincidiesen, el conjunto de discordancia es el conjunto vacío.
- Distinguimos tres tipos:
	- **Tipo 1**: Ninguno de los términos del conjunto de discordancia es un símbolo de variable.
	- **Tipo 2**: Uno de los términos es un símbolo de variable, y en el otro aparece este símbolo de variable.
	- **Tipo 3**: Uno de los términos es un símbolo de variable, y éste no aparece en el otro término.

### Algoritmo
- **Entrada**: Conjunto $W$ formado por dos o más literales.
- **Salida**: Un unificador principal (si existe), o la respuesta *no son unificables*.
- **Descripción**:
	- $W_0:=W$.
	- $\sigma:=Id$.
	- Mientras $|W_k|\geq 2$
		- Tomamos $D$ el conjunto de discordancia de $W_k$.
			- Si $D$ es de Tipo 1 o Tipo 2:
				- Devuelve: No son unificables.
				- Fin.
			- Si $D$ es de Tipo 3, de la forma $D=\{x_k,t_k\}$:
				- $\tau:=(x_k|t_k)$.
				- $W_{k+1}:=\{\tau(\alpha)\in W_k\}$.
				- $\sigma:=\tau\sigma$.
	- Devuelve $\sigma$.
	- Fin.

### Ejemplo
- Utiliza el algoritmo de unificación para encontrar un unificador principal para $\lambda_1=P(a,x,f(g(y)))$ y $\lambda_2=P(z,f(z),f(u))$.

|$D$|$\tau$|$W_i$|$\sigma$|$\lvert W_i\lvert$|
|:-:|:-:|:-:|:-:|:-:|
|||$\{P(a,x,f(g(y))), P(z,f(z),f(u))\}$|$Id$|$2$|
|$\{a,z\}$|$(z\lvert a)$|$\{P(a,x,f(g(y))), P(a,f(a),f(u))\}$|$(z\lvert a)$|$2$|
|$\{x,f(a)\}$|$(x\lvert f(a))$|$\{P(a,f(a),f(g(y))), P(a,f(a),f(u))\}$|$(x\lvert f(a))(z\lvert a)$|$2$|
|$\{g(y),u\}$|$(u\lvert g(y))$|$\{P(a,f(a),f(g(y)))\}$|$(u\lvert g(y))(x\lvert f(a))(z\lvert a)$|$1$|

- Los literales son unificables y el unificador es $\sigma=(u\lvert g(y))(x\lvert f(a))(z\lvert a)$.
- Cada fila es una iteración del algoritmo.

## Sistemas de ecuaciones en términos
- Supongamos que tenemos dos fórmulas atómicas $\alpha$ y $\beta$, que queremos ver si son unificables o no unificables. Supogamos que $\alpha=R(t_1,t_2,...,t_n)$ y $\beta=R^\prime(t_1^\prime,t_2^\prime,...,t_n^\prime)$.Obviamente, si las fórmulas empiezan por distintos símbolos de predicado, entonces no son unificables. 
- Lo que buscamos hacer es tranformar cada $t_i$ en $t_i^\prime$. Esto se puede representar como un sistema de $n$ ecuaciones donde cada expresión $t_i=t_i^\prime$ es una ecuación del sistema $e_i$. Un sistema de ecuaciones es por tanto un conjunto de ecuaciones $E=\{e_1,e_2,...,e_n\}$.
- Una solución al sistema es una sustitución $\sigma$ de forma que cualquier solución del sistema $\tau$ se puede expresar como $\tau=\tau_1\sigma$ para alguna sustitución $\tau_1$. Dicha solución es una **solución principal**.
- Dos sistemas de ecuaciones en términos son equivalentes si tienen las mismas soluciones. 
- Vamos a dar un método que dado un sistema de ecuaciones en términos, decidir si tiene o no solución, y en caso de tenerla, encontrarlas todas.

### Definición 6
- Dado un sistema de ecuaciones en términos $E=\{e_1,e_2,...,e_n\}$, se dice que está en forma resuelta si:
	- Cada ecuación del sistema es de la forma $x_i=t_i$, con $x_i$ una variable.
	- Las variables $x_1,x_2,...,x_n$ son todas distintas.
	- En los términos $t_1,t_2,...,t_n$ no hay ninguna ocurrencia de las variables $x_1,x_2,...,x_n$.
- En tal caso, $\sigma_E=(x_1|t_1;x_2|t_2;...;x_n|t_n)$ es una solución principal del sistema.

- Vamos a ver pues como transformar un sistema (si es posible) en otro sistema equivalente y que esté en forma resuelta. Para esto, vamos a ver algunos resultados que, bien permiten transformar un sistema en otro equivalente o bien concluyen que el sistema no tiene solución.
	- Los sistemas $E\cup\{t=t\}$ y $E$ son equivalentes. Es decir, podemos eliminar las ecuaciones en las que el término de la derecha y el de la izquierda coinciden.
	- Los sistemas $E\cup\{t_1=t_2\}$ y $E\cup\{t_2=t_1\}$ son equivalentes.
	- Los sistemas $E\cup\{f(t_1,...,t_m)=f(t_1^\prime,...,t_m^\prime)\}$ y $E\cup\{t_1=t_1^\prime,...,t_m=t_m^\prime\}$ son equivalentes.
	- Los sistemas $E\cup\{x=t\}$, donde $x$ es una variable que no aparece en el término $t$, y $\sigma(E)\cup\{x=t\}$ son equivalentes, donde $\sigma$ es la sustitución $\sigma=(x|t)$. En este caso, $\sigma(E)$ es el sistema formado por las ecuaciones que resultan de realizar la sustitución $\sigma$ en cada uno de los términos que intervienen en el sistema $E$.
	- Un sistema de la forma $E\cup\{x=t\}$ donde $t$ es un término distinto de $x$, pero en el que interviene la variable $x$, no tiene solución.
	- Un sistema de la forma $E\cup\{f(t_1,...,t_n)=g(t_1^\prime,...,t_m^\prime)\}$ con $f$ y $g$ símbolos de función distintos, no tiene solución.
	-  Un sistema de la forma $E\cup\{f(t_1,...,t_n)=a\}$, o de la forma $E\cup\{a=b\}$, donde $a$ y $b$ son constantes distintas, no tiene solución.

### Ejemplo
- Estudia si el siguiente conjunto de fórmulas es unificable:
	- $\{P(x,y), P(f(z),x), P(u,f(x))\}$
	
	- $(1)\begin{cases} x=f(z) \\ y=x \\ x=u \\ y=f(x) \end{cases}$
	
	-  $(2) \begin{cases} x=f(z) \\ y=f(z) \\ x=u \\ y=f(x) \end{cases}$

	- $(3) \begin{cases} x=f(z) \\ y=f(z) \\ u=f(z) \\ y=f(x) \end{cases}$

	- $(4) \begin{cases} x=f(z) \\ y=f(z) \\ u=f(z) \\ f(z)=f(f(z)) \end{cases}$

	- $(5) \begin{cases} x=f(z) \\ y=f(z) \\ u=f(z) \\ f(z)=f(f(z)) \end{cases}$

	- $(6) \begin{cases} x=f(z) \\ y=f(z) \\ u=f(z) \\ z=f(z) \end{cases}$

- Al quedarnos la ecuación $z=f(z)$ vemos que no es unificable.
- Otro ejemplo:

![[ecuation-system-unification-example.png]]

# Resolución 4.2
- El problema que teníamos con la resolución dentro de la [[first order languages|lógica de primer orden]] era la aparición de variables, lo que hacía que se pudiese tomar cualquier valor, es decir, podría ser cualquier término. Una vez explicado el método por el cuál podemos hacer sustituciones para sustituir las variables por términos podemos continuar con el [[propositional logic#El problema de la implicación semántica 2.6|problema de la implicación semántica]] y su resolución por medio del método de [[propositional logic#Método de resolución 2.6.2|resolución]].
- Pero antes de ello, debemos adaptar ciertos conceptos al lenguaje de primer orden.

## Resolventes 4.2.1
### Resolventes binarias
### Definición 7
- Sean $C_1$ y $C_2$ dos cláusulas que no tienen variables comunes. Supongamos que hay dos literales unificables $L_1$ y $L_2$ tales que $C_1=L_1\lor C_1^\prime$ y $C_2=L_2^c\lor C_2^\prime$, donde $C_1$ y $C_2$ son cláusulas (que podríam ser la cláusula vacía). Sea $\sigma$ un unificador de $L_1$ y $L_2$.
- Entonces, la cláusula $\sigma(C_1^\prime)\lor\sigma(C_2^\prime)$ es una resolvente binaria de $C_1$ y $C_2$.
- Por ejemplo, $Q(b)$ es una resolvente binaria de las cláusulas $\neg P(x)\lor Q(b)$ y $P(a)$.
- Otro ejemplo, $Q(a,a)\lor R(b)$ es una resolvente binaria de $P(x,b)\lor Q(x,a)$ y $\neg P(a,z)\lor R(z)$.
- Una vez con el conocimiento suficiente como para poder obtener resolventes, sabemos que si de un conjunto de cláusulas podemos obtener la cláusula vacía por medio de resolventes, podemos decir que ese conjunto es insatisfacible.

![[binary-resolvent-example.png]]

### Resolventes
- Pese a ello, las resolventes binarias no son suficiente para abarcar todos los conjuntos. Hay algunos con los que nunca podríamos obtener la cláusula vacía, pese a ser conjuntos insatisfacibles.

### Definición 8
- Sea $C$ una cláusula. Supongamos que en $C$ hay dos o más literales que son unificables, y sea $\sigma$ un unificador. En ese caso, diremos que $\sigma(C)$ es un factor de $C$.
- En el caso de que $\sigma(C)$ sea un factor de $C$, se verifica que $C\models \sigma(C)$.
- Por ejemplo, sea $C=P(x)\lor P(f(a))\lor Q(x,b)$, entonces $P(f(a))\lor Q(f(a),b)$ es un factor de $C$ (hemos unificado los dos primeros literales con $(x|f(a))$).
- Dentro de una cláusula, no podemos renombrar variables con varias ocurrencias para unificar los literales. Si fuesen cláusulas distintas sí.

### Definición 9
- Sean $C_1$ y $C_2$ dos cláusulas. Se dice que $C$ es una resolvente de $C_1$ y $C_2$ si $C$ responde a alguna de las cuatro posibilidades siguientes:
	- $C$ es una resolvente binaria de $C_1$ y $C_2$.
	- $C$ es una resolvente binaria de $C_1$ y un factor de $C_2$.
	- $C$ es una resolvente binaria de un factor de $C_1$ y de $C_2$.
	- $C$ es una resolvente binaria de un factor de $C_1$ y de un factor de $C_2$.

## Deducciones y principio de resolución 4.2.2
- Sea $\Gamma$ un conjunto de cláusulas, y $C$ una cláusula. Una *deducción* de $C$ a partir de $\Gamma$  es una  sucesión finita de cláusulas $C_1,C_2,...,C_n$ donde $C_n=C$, y para $i<n$, $C_i$ es una cláusula, que bien es un elemento de $\Gamma$, bien es un resolvente de dos cláusulas que la preceden. 

## Teorema 4.2.1 
- (Completitud del principio de resolución)
- Sea $\Gamma$ un conjunto de cláusulas. Entonces $\Gamma$ es insatisfacible si, y sólo si, existe una deducción de $\square$ a partir de $\Gamma$.

![[resolution-example-first-order-language.png]]

# Estrategias de gestión 4.3
## Estrategia de saturación 4.3.1
- Consiste en el cálculo de todas las posibles resolventes. El procedimiento puede ejecutarse de forma algorítmica:
	- Llamamos $S_0=S$.
	- Para cada $i$
		- Calculamos $S_{i+1}$ como el conjunto que se obtiene de $S_i$ al añadir todas las resolventes que puedan calcularse usando clásulas de $S_i$.
		- Si $S_{i+1}$ contiene a la cláusula vacía, entonces  el conjunto de partida es *insatisfacible*.
		- Si $S_{i+1}=S_i$, entonces el conjunto de partida es *satisfacible*.
		- En otro caso, incrementar $i$ y volver al inicio.
- Este algoritmo nos proporciona una cadena de conjuntos de cláusulas, es decir, una secuencia de conjuntos en el que cada uno está contenido en el siguiente:
	- $S_0\subseteq S_1\subseteq ...\subseteq S_i\subseteq S_i+1\subseteq ...$
- Todos tienen el mismo carácter de satisfacibilidad, si uno lo es, lo deben ser los demás.
- Terminamos cuando la cadena se estabiliza ($S_i=S_{i+1}$).

## Deducciones lineales 4.3.2
- [[propositional logic#Resolución lineal|Deducción lineal]].
- [[propositional logic#Estrategia Input|Estrategia input]].
- [[propositional logic#Definición 12|Conjuntos de Horn]].
