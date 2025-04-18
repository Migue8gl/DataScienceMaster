- #inductionandrecurrence | #inducciónyrecurrencia | #lmd

# Inducción y deducción 5.1
- El proceso por el cual inferimos la certeza de una proposición particular a partir de la certeza de una proposición general se le conoce como #deducción.
- Recíprocamente a la inferencia de la certeza de una proposición general a partir de la certeza de una o varias proposiciones particulares, se le conoce como #inducción.

# Números naturales. Principio de inducción 5.2
- $\mathbb{N}=\{0,1,2,...\}$.
- Para llegar a este conjunto podemos basarnos en los axioamas de #peano, o bien, partir de la construcción basada en la teoría de conjuntos de Zermelo-Fraenkel, y de la que los axiomas de Peano son consecuencias de la construcción realizada.
- Dado dos números naturales $m$ y $n$ podemos definir otros dos números naturales, suma y producto, que se representan como $m+n$ y $m·n$ (o simplemente $mn$). Esto nos define dos operaciones en $\mathbb{N}$, que satisfacen las siguientes propiedades.
	- Para cualesquiera $m,n,p\in\mathbb{N}, (m+n)+p=m+(n+p)$ (es decir, la suma es asociativa).
	- Para cualesquiera $m,n,p\in\mathbb{N},m+n=n+m$ (es decir, la suma es conmutativa).
	- Existe en $\mathbb{N}$ un elemento, representado por $0$ tal que para cada $m\in\mathbb{N}$ se tiene que $m+0=m$ (existencia de elemento neutro para la suma).
	- Si $m+n=m+p$ entonces $n=p$ (propiedad cancelativa).
	- Para cualesquiera $m,n,p\in\mathbb{N}, (m·n)·p=m·(n·p)$ (es decir, el producto es asociativo).
	- Para cualesquiera $m,n,p\in\mathbb{N},m·n=n·m$ (es decir, el producto es conmutativo).
	- Existe en $\mathbb{N}$ un elemento, representado por $1$ tal que para cada $m\in\mathbb{N}$ se tiene que $m·1=m$ (existencia de elemento neutro para el producto).
	-  Si $m·n=m·p$ y $m\neq 0$ entonces $n=p$.
	- Para cualesquiera $m,n,p\in\mathbb{N}, m·(n+p)=m·n+m·p$ (es decir, la suma es distributiva respecto al producto).
	- Al conjunto que resulta de quitar el $0$ de $\mathbb{N}$ se le denota como $\mathbb{N^*}$.
- También en $\mathbb{N}$ hay definida una relación que se denomina $\leq$. Es decir, dados $m,n\in\mathbb{N}$ tenemos un criterio para decir si $m\leq n$ es verdadero o es falso. Esta relación satisface las siguientes propiedades:
	- $m\leq m$ para todo $m\in\mathbb{N}$ (reflexiva).
	- Si $m\leq n$ y $n\leq m$ entonces $m=n$ (antisimétrica).
	- Si $m\leq n$ y $n\leq p$ entonces $m\leq p$ (transitiva).
	- Para cualesquiera $m,n\in\mathbb{N}, m\leq n$ ó $n\leq m$.
- Una relación que satisface las tres primeras propiedades se conoce como relación de orden. Si además satisface la cuarta, tenemos un orden total.
	- $m\leq n$ implica que $m+p\leq n+p$ para todo $p\in\mathbb{N}$.
	- $m+p\leq n+p$ implica que $m\leq n$.
	- $m\leq n$ implica que $m·p\leq n·p$.
	- Si $m·p\leq n·p$ y $p\neq0$ entonces $m\leq n$.

### Principio de inducción
- Si $A$ es un subconjunto de $\mathbb{N}$ tal que:
	- $0\in A$
	- Si $n\in A$ entonces $n+1\in A$
- Entonces $A=\mathbb{N}$.
- Es decir, cualquier número natural puede ser obtenido a partir del cero sin más que sumar uno las veces necesarias. Para otros conjuntos, como $\mathbb{Q}$ esto no es así.
- Si queremos demostrar que $P(n)$ es cierto para todo $n\in\mathbb{N}$ (donde $P(n)$ es una propiedad que hace referencia a $n$), podemos hacerlo como sigue:
	- Caso base: Demostramos que $P(0)$ es cierto.
	- Hipótesis de inducción: Suponemos que $P(n)$ es cierto.
	- Paso inductivo: A partir de la hipótesis de inducción, demostramos que es cierto $P(n+1)$.

#### Ejemplo 1
- Vamos a comprobar que para todo $n\geq 1$ se verifica que:
	- $1+2+...+n=\frac{n(n+1)}{2}$
- Hacemos la demostración por inducción:
	- Caso base: Para $n=1$ el resultado es trivialmente cierto.
	- Hipótesis de inducción: Para un número natural $n$ se tiene que $1+2+...+n=\frac{n(n+1)}{2}$.
	- Paso inductivo: A partir de la hipótesis de inducción hemos de probar que:
		- $1+2+...+n+(n+1)=\frac{(n+1)(n+2)}{2}$
	- Entonces
		- $1+2+...+n+(n+1)=\frac{n(n+1)}{2}+n+1=$
		- $\frac{n(n+1)}{2}+\frac{2(n+1)}{2}=$
		- $\frac{n(n+1)+2(n+1)}{2}=$
		- $\frac{n²+n+2n+2}{2}=$
		- $\frac{n²+3n+2}{2}=$
		- $\frac{(n+1)(n+2)}{2}$

#### Ejemplo 2
- Vamos a demostrar que para cualquier número natural $n$, el número $7^n-1$ es múltiplo de 6.
	- Caso base: Para $n=0$ que es el primer elemento, vemos que se cumple, pues $7⁰-1=0$ y $0$ es múltiplo de $6$.
	- Hipótesis de inducción: Para un número natural $n$ se tiene que que $7^n-1$ es múltiplo de 6.
	- Paso inductivo: A partir de la hipótesis de inducción debemos demostrar que:
		- $7^{n+1}-1=6k$
	- Entonces:
		- Por la hipótesis de inducción sabemos que $7^n=6k+1$, por lo que $7^{n+1}-1=$
		- $7·7^n-1=$
		- $7·(6k+1)-1=$
		- $7·6k+6=$
		- $6·(7k+1)$, lo que es claramente un múltiplo de $6$.
- El principio de inducción nos dice que si $A$ es un subconjunto de $\mathbb{N}$ que satisface las dos siguientes propiedades:
	- $0\in A$
	- $n\in A\implies n+1\in A$
- Entonces $A=\mathbb{N}$.

### Teorema 5.2.1
- (Principio de buena ordenación) Sea $A$ un subconjunto de $\mathbb{N}$ distinto del conjunto vacío. Entonces $A$ tiene mínimo.

### Teorema 5.2.2
- (Segundo principio de inducción) Sea $A$ un subconjunto de $\mathbb{N}$. Supongamos que se verifica:
	- $0\in A$.
	- Para cualquier $n,\{0,1,...,n-1\}\subseteq A\implies n\in A$.
- Entonces $A=\mathbb{N}$.

# Recurrencia 5.3
## Definiciones recursivas 5.3.1
### Definición 1
- Sea $X$ un conjunto. Una sucesión en $X$ es una aplicación $x:\mathbb{N}\rightarrow X$.
- Si $x:\mathbb{N}\rightarrow X$ es una sucesión, denotaremos al elemento $x(n)$ como $x_n$.
- A la hora de definir una sucesión en $X$ podemos definir una regla que asigne a cada número $n$ un elemento $x_n$. Como por ejemplo, $x_n=n-n^2$. En ocasiones, podemos definir la imagen de un número natural $n$ basándonos en la imagen de otro número natural, de forma que para definir una sucesión recurrimos a la propia sucesión. A este proceso se le conoce como *recursión*. 
- Para definir una sucesión recursiva $x:\mathbb{N}\rightarrow X$ seguimos los siguientes pasos:
	- Paso base: Se elige un elemento $x_0\in X$, tal que ese valor en la sucesión será $n=0$.
	- Paso recursivo: Dado un número natural $n$, se proporciona una regla para definir $x_{n+1}$ a partir del conocimiento de $x_n$.
- Una segunda forma sería:
	- Paso base: Se eligen elementos $x_0,x_1,...,x_{k-1}\in X$.
	- Paso recursivo: Dado un número natural $n\geq k$, se proporciona una regla para definir $x_n$ a partir del conocimiento de $x_{n-1},x_{n-2},...,x_{n-k}$.
- Por ejemplo definimos la siguiente sucesión:
	- $f_0=0;f_1=1$.
	- $f_n=f_{n-1}+f_{n-2}$ si $n\geq 2$.
	- $0,1,1,2,3,5,8,13...$ (Sucesión de Fibonacci).
- Este tipo de sucesión satisface lo que se conoce como *relación de recurrencia homogénea*.

## Recurrencia lineal homogénea 5.3.2
### Definición 2
- Sea $x:\mathbb{N}\rightarrow\mathbb{R}$ una sucesión. Decimos que dicha sucesión satisface una relación de recurrencia lineal homogénea con coeficientes constantes si existe $k\in\mathbb{N}$ y $a_1,...,a_k\in\mathbb{R}$ tal que para cualquier $n\geq k$ se verifica que:
	- $\sum_{j=0}^ka_j·x_{n-j}=a_0·x_n+a_1·x_{n-1}+...+a_k·x_{n-k}=0$
- Donde $a_0=1$.
- Al número $k$ se le denomina orden de la relación. A una sucesión que satisface una relación de recurrencia lineal homogénea la llamaremos sucesión lineal homogénea.
- Es lineal pues el exponente de todos los términos es $1$. Se llama homogénea pues está igualada a cero. Se dicen que son coeficientes constantes pues lo que multiplica a los términos de la sucesión son números (siempre los mismos).
- En general, si $x_n$ es una sucesión que satisface una relación de recurrencia lineal homogénea de orden $k$, entonces para cualquier $m\geq k$, $x_n$ satisface una relación de recurrencia lineal homogénea de orden $m$.
- Llamaremos grado de una sucesión que satisfaga una relación de recurrencia lineal homogénea al menor orden de las relaciones de recurrencia lineal homogénea que satisface la sucesión.
- Si tenemos dos sucesiones $x_n, y_n$ que satisfacen una relación de este tipo, entonces para cualesquiera $a,b\in\mathbb{R}$, la sucesión $z_n=ax_n+by_n$ también la satisface.
- Si tenemos una sucesión lineal homogénea de orden $k$, entonces para calcular cada término de la sucesión necesitamos conocer los $k$ términos anteriores. Estos primeros $k$ términos se conocen como **condiciones iniciales**.

### Ejemplo
- Vamos a buscar la solución al problema de recurrencia lineal homogénea $x_n=x_{n-1}+2x_{n-2}$ con condiciones iniciales de $x_0=2$ y $x_1=1$. Para esto, calculamos unos cuantos términos de dicha sucesión:
	- Para $n\geq 2\rightarrow 5,7,17,31,65,...$
	- Vemos que $x_n\rightarrow 2,1,5,7,17,31,65$
	- Si la comparamos con la sucesión $2^n\rightarrow 1,2,4,8,16,32$
	- Los términos difieren en $-1$ o $+1$, por lo que una posible sucesión sería $2^n+(-1)^n$.
	- Vamos a comprobar que eso es cierto, para eso, utilizaremos el [[induction and recurrence#Teorema 5.2.2|segundo principio de inducción]].
	- Para $n=0, n=1$ sabemos que es cierto.
	- Sea $n\geq 2$ y suponemos que para cualquier $k<n$ se verifica que $x_k=2^k+(-1)^k$.
		- $x_n=x_{n-1}+2x_{n-2}=2^{n-1}+(-1)^{n-1}+2(2^{n-2}+(-1)^{n-2})$
		- $=2^{n-1}+(-1)^{n-1}+2·2^{n-2}+2·(-1)^{n-2}$
		- $=2^{n-1}+2^{n-1}+(-1)^{n-2}·(-1+2)$
		- $=2^n+(-1)^n$.
- De aquí podemos concluir que el término general de la sucesión es $x_n=2^n+(-1)^n$.

### Definición 3
- Dado el problema de recurrencia lineal homogénea con coeficientes constantes.
	- $x_n+a_1x_{n-1}+...+a_kx_{n-k}=0$
- Al polinomio $x^k+a_1x^{k-1}+...+a_{k-1}x+a_k$ se le conoce como el polinomio característico de la relación, y a la ecuación $x^k+a_1x^{k-1}+...+a_{k-1}x+a_k=0$ como la ecuación característica.

### Proposición 5.3.1
- Si $\alpha$ es una solución de la ecuación característica de un problema de recurrencia, entonces la sucesión $x_n=\alpha^n$ es una solución a dicho problema.
- En general, si $\alpha_1,\alpha_2,...,\alpha_k$ son todas las raíces del polinomio característico de una relación de recurrencia, entonces cualquier sucesión de la forma $x_n=b_1(\alpha_1)^n+b_2(\alpha_2)^n+...+b_k(\alpha_k)^n$ es solución a la relación de recurrencia.
- Cada una de las soluciones $\alpha$ son linealmente independientes y forman una base del subespacio vectorial de soluciones. Las condiciones iniciales son las que determinan los valores de los coeficientes $b$.

### Ejemplo
- Consideramos la sucesión definida por $x_n=x_{n-1}+2x_{n-2},x_0=2,x_1=1$. Vamos a hallar el término general de esta sucesión.
	- $x_n-x_{n-1}-2x_{n-2}=0$.
	- Calculamos el polinomio característico: $x^2-x-2$.
	- Calculamos sus raíces, que son $x\in\{2,-1\}$.
	- Una solución de la recurrencia es del tipo $b_0(2)^n+b_1(-1)^n$. Con las condiciones iniciales dadas, podemos deducir lo siguiente:
		- $x_0=2$ -> $b_0(2)^0+b_1(-1)^0=2=b_0+b_1$
		- $x_1=1$ -> $b_0(2)¹+b_1(-1)^1=1=2b_0-b_1$.
	- Resolvemos el sistema, que tiene soluciones $b_0=1, b_1=1$.
	- Por ello, la solución es $x_n=2^n+(-1)^n$.

### Proposición 5.3.2
- Supongamos que tenemos el problema de recurrencia $x_n+a_1x_{n-1}+...+a_kx^{n-k}$, que $p(x)$ es el polinomio característico de esa relación y que $\alpha$ es una raíz doble de dicho polinomio. Entonces la sucesión $x_n=n·\alpha^n$ es una solución a dicho problema.

### Ejemplo
- Consideramos la sucesión definida por la recurrencia $x_n=3x_{n-1}-3x_{n-2}+x_{n-3}$ con condiciones $x_0=4,x_1=2,x_2=2$.
- El polinomio característico es $x³-3x^2+3x-1$ que es los mismo que $(x-1)^3$, por lo que sus raíces son $x=1$.
- Una solución general es entonces $x_n=a(1^n)+bn(1^n)+cn²(1^n)=a+bn+cn^2$.
- La solución es $a=4,b=-3,c=1$ (tenemos que hacer un sistema de ecuaciones con las condiciones iniciales). Por tanto la sucesión es $x_n=n²-3n+4$.

## Recurrencia lineal no homogénea 5.3.3
### Definición 4
- Sea $x:\mathbb{N}\rightarrow\mathbb{R}$ una sucesión. Decimos que dicha sucesión satisface una relación de recurrencia lineal con coeficientes constantes si existe $k\in\mathbb{N},a_1,...,a_k\in\mathbb{R}$ y $f:\mathbb{N}\rightarrow\mathbb{R}$ tales que para cualquier $n\geq k$ se verifica que:
	- $\sum_{j=0}^ka_j·x_{n-j}=a_0·x_n+a_1·x_{n-1}+...+a_k·x_{n-k}=f(n)$
- Donde $a_0=1$ y $k$ es el orden de la relación.
- Al problema de recurrencia lineal homogénea $x_n+a_1·x_{n-1}+...+a_k·x_{n-k}=0$ lo llamaremos *problema de recurrencia lineal homogénea asociada*.

### Proposición 5.3.3
- Sea $x_n+a_1·x_{n-1}+...+a_k·x_{n-k}=f(n)$ un problema de recurrencia lineal no homogénea.
	- Supongamos que $u_n$ y $y_n$ son dos soluciones a dicho problema. Entonces la sucesión $u_n-y_n$ es una solución al problema de recurrencia lineal homogénea asociado.
	- Si $y_n$ es una solución al problema no homogéneo, entonces todas las soluciones a dicho problema son de la forma $y_n+h_n$, donde $h_n$, es una solución al problema homogéneo. 

### Proposición 5.3.4
- Supongamos que $x_n$ es una sucesión que satisface una relación de recurrencia lineal no homogénea.
	- $x_n+a_1x_{n-1}+...+a_kx_{n-k}=f(n)$
- Donde $f(n)$ es un polinomio de grado $r$.
- Entonces $x_n$ satisface una relación de recurrencia lineal homogénea cuyo polinomio característico es $(x^k+a_1x^{k-1}+...+a_k)(x-1)^{r+1}$.

### Proposición 5.3.5
- Supongamos que tenemos que $x_n$ es una sucesión que satisface una relación de recurrencia lineal no homogénea.
	- $x_n+a_1x_{n-1}+...+a_kx_{n-k}=b^n·f(n)$
- Donde $f(n)$ es un polinomio de grado $r$.
- Entonces $x_n$ satisface una relación de recurrencia lineal homogénea cuyo polinomio característico es $(x^k+a_1x^{k-1}+...+a_k)(x-b)^{r+1}$. En caso de tener varios términos del tipo $b_i^n·f(n)_i$ entonces el polinomio característico sería $(x^k+a_1x^{k-1}+...+a_k)(x-b_1)^{r_1+1}...(x-b_i)^{r_i+1}$.