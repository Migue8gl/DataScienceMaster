- #boolealgebra | #algebradeboole | #lmd

# Álgebras de Boole 1.1
## Generalidades de álgebras de Boole 1.1.1
### Definición 1
- Un álgebra de Boole es una seis-tupla $(A,\lor,\land,\neg,0,1)$ donde A es un conjunto no vacío, $\lor$ y $\land$ son operaciones binarias, $\neg$ es una operación monaria y 0 y 1 elementos de A. Además $\forall a, b \in A$:

![[boole-algebra-properties.png]]

### Propiedades
- Si A es un álgebra de Boole:
	- **Dominación**:  $a\lor1=1$ ; $a\land0=0$.
	- **Absorción**: $a\land(a\lor b)=a$ ; $a\lor(a\land b) = a$.
	- **Idempotencia**: $a\lor a=a$ ; $a\land a=a$.
	- **Cancelativa**: $a\lor b=a\lor c \implies b=c$ ; $a\land b=a\land c \implies b=c$.
	- **Doble complementario**: $\neg\neg a=a$.
	- **Leyes de De Morgan**: $\neg(a\lor b)=\neg a\lor\neg b$ ; $\neg(a\land b)=\neg a\land\neg b$.
	- $a\lor(\neg a\land b)=a\lor b$ ; $a\land(\neg a\lor b)=a\land b$.

### Principio de dualidad
- Dada una sentencia (enunciado) en un álgebra de Boole, entonces el resultado de intercambiar $\lor$ por $\land$ y 0 por 1 se conoce como resultado dual. Si un teorema es cierto para álgebras de Boole, su teorema dual también.

### Proposición 1.1.1
- Sean $(B_1,\lor_1,\land_1)$ y $(B_2,\lor_2,\land_2)$ dos álgebras de Boole. Entonces el conjunto $B_1XB_2$ con las operaciones $(x,y)\lor(x^`,y^`)=(x\lor_1x^`,y\lor_2y^`)$ ; $(x,y)\land(x^`,y^`)=(x\land_1x^`,y\land_2y^`)$ tiene también estructura de álgebra de Boole.
- Si hacemos el producto de $\mathbb{B}$ consigo misma obtenemos $\mathbb{B}^2$.

### Definición 2
- Sea B un álgebra de Boole, entonces definimos una relación en B como:
	- $a\leq b\iff a\lor b=b$ 
- También puede escribirse como $aRb$. 

### Teorema 1.1.1
- Sea $(B,\lor,\land)$ un álgebra de Boole y ($\leq$ o $R$) la relación definida en [[boole algebra#Definición 2|Definición 2]]. Esta es una relación de orden en B. Además dados dos elementos $a,b\in B$ se tiene  que $sup(a,b)=a\lor b$ e $inf(a,b)=a\land b$. Este conjunto tiene máximo y mínimo que son 1 y 0 respectivamente.

### Diagramas de Hasse
- Una forma de representar un conjunto ordenado es con un diagrama de #hasse, de forma que podemos representar las álgebras de Boole con estos diagramas.

![[hasse-diagram.png]]

## Estructura de álgebras de Boole finitas 1.1.2
- Si B es un álgebra de Boole finita (con elementos finitos) entonces el número de elementos de B es una potencia de 2 -> $|B|=2^n$. 
- Si B y C son dos álgebras de Boole, un isomorfismo entre ambas es una aplicación biyectiva $f:B\to C$ tal que $f(x\lor y)=f(x)\lor f(y)$ y  $f(x\land y)=f(x)\land f(y)$.
- Si dos álgebras de Boole finitas tienen el mismo número de elementos entonces son isomorfas.

### Definición 3
- Sea B un álgebra de Boole y $x\in B$. Se dice que x es un átomo de x es un elemento minimal de $B\backslash \{0\}$. Es decir, minimal dentro de los positivos.
- Notaremos $At(A)$ al conjunto de átomos del álgebra de Boole A. 
- Un átomo de un subálgebra puede no ser un átomo del álgebra, pero todos los átomos del álgebra si lo serán de sus subálgebras.

### Teorema 1.1.2
- Sea B un álgebra de Boole finita y $x\in B\backslash \{0\}$. Entonces x se expresa de forma única como supremo de átomos.
- Supremo recordemos que es $x\lor y$.

### Lema 1.1.1
- Sea B un álgebra de Boole finita y $x\in B\backslash \{0\}$. Entonces existe $a\in B$, átomo, tal que $a\leq x$.


# Funciones Booleanas 1.2
## Definición funciones booleanas 1.2.1
- El conjunto de funciones booleanas de $n$ variables tiene estructura de álgebra de Boole.
- Los átomos de este tipo especial de álgebra se denominan #minterms o minitérminos -> [[boole algebra#Corolario 1.2.1|corolario 1.2.1]].
- El [[boole algebra#Teorema 1.1.2| teorema 1.1.2]] aplicado a esta álgebra nos dará la denominada formal normal canónica [[disjunction|disyuntiva]] #fcd. 
- Cambiamos la notación de $\lor,\land$ por $+,\cdot$ , pero misma notación para el complementario.
- No confundir suma y producto en $\mathbb{Z}_2$ con la suma y producto booleano.

### Definición 4
- Una función booleana (o función booleana elemental) en n variables es una aplicación $f:\mathbb{B}^n\to\mathbb{B}$. A partir de ahora $F_n$ es el conjunto de funciones booleanas con $n$ variables. 
- Por ejemplo, $F_1$ son las funciones de $1$ variable.
- Los átomos de este álgebra de Boole son las aplicaciones que valen $1$ en un elemento y $0$ en el resto.
- $F_n$ tiene $2^n$ átomos, por lo que tiene $2^{2^n}$ elementos. Ya que en $\mathbb{B}^n$ hay $2^n$ elementos -> [[boole algebra#Estructura de álgebras de Boole finitas 1.1.2|estructuras de álgebras de Boole finitas]].

## Expresiones booleanas 1.2.2
### Definición 5
- Sea S un conjunto. Se definen las expresiones booleanas sobre el conjunto S de forma recursiva como sigue:
	- Si $x\in S\cup\{0,1\}$ entonces x es una expresión booleana.
	- Si $e_1$, $e_2$ son dos expresiones booleanas, entonces $e_1+e_2, e_1\cdot e_2,\neg e_1$ son expresiones booleanas.
- Las expresiones de S, o complementos suyos, los llamaremos literales.

### Expresiones equivalentes

![[equivalent-boolean-functions.png]]

### Definición 6
- Los #minterms o minitérminos en n variables son productos de n literales, cada uno con una variable diferente.
- Ejemplo: Sea $S=\{x,y,z\}$, entonces son minterms $xyz, \neg x,y,z, \neg x,\neg y,z$, pero no lo son $xy, xyx, \neg x,y,x$.

### Lema 1.2.1
- Sea $m$ un minterm en $n$ variables. Entonces $m$, por ser una expresión booleana, determina una función booleana $f:\mathbb{B}^n\to\mathbb{B}$. Esta función vale $1$ en un elemento y $0$ en el resto.

### Corolario 1.2.1
- Los minterm son átomos del álgebra $F_n$.

### Corolario 1.2.2
- Toda función booleana se expresa de forma única, salvo orden, como suma (supremo) de minterm -> [[boole algebra#Teorema 1.1.2|teorema 1.1.2]].

### Conseguir forma normal disyuntiva
- Para ello partimos de las [[boole algebra#Expresiones equivalentes|equivalencias]] anteriores. A partir de estas y una función booleana, hacemos que esta se quede como suma de minterms.

![[fcd-algorithm.png]]

### Numeración
- Cada elemento de $\mathbb{B}^n$ es una secuencia de $n$ dígitos de $0s$ y $1s$. De forma que cada elemento es la expresión en binario de un número entre $[0,2^n-1]$. A cada elemento de $\mathbb{B}^n$ le corresponde un minterm.
- De esta forma a cada minterm podemos numerarlo por su correspondiente número en binario. Es decir, el minterm $xy\overline z\overline t$ toma el valor $1$ en $(1,1,0,0)$ de forma que este minterm es el $m_{12}$, ya que $1100$ es $12$ en binario.

### Resultado dual de forma canónica disyuntiva
- El resultado dual de expresar una función booleana como suma de minterms es expresarla como suma de #maxterms.
- Un maxterm en $n$ variables es una suma de $n$ literales en los que no aparece ninguna variable repetida. Se corresponde con una función booleana en la que todos los elementos  dan resultado $1$ excepto uno, que da $0$.
- Se expresan con $M$ mayúscula, en contraposición con el minterm que se expresa con $m$ minúscula.
- Esta forma es la forma canónica [[conjunction|conjuntiva]], #fcc.

## Puertas lógicas 1.2.3
- Las funciones booleanas nos proporcionan un modelo matemático para el diseño de circuitos electrónicos, pues estos reciben entradas booleanas $(0,1)$ y producen una salida también binaria.

### Puerta NOT
- Cambia la entrada por su complementario.

![[not-logic-door.png]]

### Puerta AND y OR
- La puerta #and toma como entrada dos o más variables y devuelve el producto de estas. La puerta #or devuelve la suma.

![[logic-doors.png]]

## Optimización de funciones booleanas 1.2.4
### Definición 7
- Un #implicante de una función es un [[monomial]] (monomio) fundamental que es menor o igual que la función.
- Un implicante primo es un implicante de la función en el que al suprimir un literal deja de ser implicante.

### Teorema 1.2.1
- La suma de todos los implicantes primos de una función es una forma [[disjunction|disyuntiva]] de la función.

### Mapas de Karnaugh
- Los mapas de Karnaugh son una disposición plana en forma de cuadrícula de la tabla de una función booleana que nos ayudará a obtener los implicantes primos y las formas no redundantes de la función.

![[karnaugh-map.png]]

- Hay que encontrar minitérminos que se diferencien solo en un literal. En tal caso ambos minterms quedan reducidos a uno solo (que deja de ser un minterm, pues le falta una variable).
- En un mapa de #karnaugh cada celda representa un minterm de todos los posibles ($8$ minterm para $3$ variables por ejemplo).
- Dos celdas adyacentes se diferencian en un solo minterm, así como dos celdas opuestas en una fila o columna.
- El objetivo es agrupar "unos" en el menor número posible de bloques y de mayor tamaño (que debe ser potencia de $2$).

![[karnaugh-example.png]]

- El cuarto mapa daría como resultado la función $f(x,y,z,t)=\overline y\overline z+yz$, ya que en el bloque verde solo se dejan la $y$ siendo $1$, que coincide en ambos minterms, y la $z$ que también es $1$. En el bloque rojo se repiten estos literales, pero siendo cero, por lo que se niegan.
- También se puede aplicar el mapa partiendo de formas normales [[conjunction|conjuntivas]], #fcc, en vez de marcar en el mapa los minterms donde la función es $1$, se marcan los maxterm donde la función es $0$.
- El resultado nos da la forma reducida.

### Método de Quine-McCluskey
- Los mapas de Karnaugh son una forma de reducir las funciones booleanas, pero dado a que es un método que utiliza un diagrama, su uso para funciones de más de cuatro variables es inadecuado e ineficiente.
- Con este método se determina en primera instancia que términos son candidatos a aparecer en el desarrollo minimal, después se seleccionan de esos candidatos los que intervienen.
- Sabemos que cada minterm en $n$ variables va asociado a una cadena de $n$ bits. Debemos pues ordenar esas cadenas de bits en una columna, agrupando aquellas con igual número de "unos".
- Comparamos las cadenas de un grupo con aquellas de otro grupo inmediatamente inferior. Si encontramos dos cadenas que se diferencien en un solo bit, las marcamos y, en una columna situada a la derecha, representamos estas dos cadenas por una nueva en la que se sustituye ese bit por un "-". Si aparecen dos cadena iguales dejamos una
- Una vez hecho esto nos vamos a la siguiente columna y repetimos hasta que no se pueda formar otra columna.
- Se seleccionan aquellas cadenas no marcadas.
- Cada una de las cadenas finales se corresponde con un término que es producto de literales, un implicante primo -> [[boole algebra#Definición 7|implicantes]].
- Ej:

![[quine-example.png]]

- La segunda parte del método consiste en encontrar, de entre todos los implicantes primos, el menor conjunto de ellos que represente a la expresión booleana dada.
- Para ello hacemos una tabla cuyo eje horizontal son los minterms de la función booleana y cuyo eje horizontal son los implicantes primos hallados.
- Señalamos aquellas celdas cuyos minterm contengan al implicante que estemos analizando, diremos que el implicante primo **cubre** al minterm.
- Elegimos los implicantes primos esenciales, estos son aquellos para los que un minterm solo queda cubierto por ellos mismos y ningún otro implicante. Columnas marcadas solo por un implicante.
- Eliminamos esos implicantes y sus respectivas columnas, buscamos de nuevo implicantes primos y repetimos el proceso hasta que no haya más.
- Si no hay implicantes esenciales, buscamos implicantes intercambiables (cubren a los mismos minterm) y borramos todos excepto uno.
- Si no hay tampoco intercambiables buscamos implicantes dominantes (**a** domina a **b** si todos los minterms cubiertos por **b** están cubiertos también por **a**) y borramos el minterm dominado.
- En caso de llegar a una tabla sin implicantes primos esenciales ni dominantes usamos algoritmo de #petrick.

### Método Petrick
- Escogemos todos los implicantes primos que hemos obtenido con el algoritmo de [[boole algebra#Método de Quine-McCluskey|Quine-McCluskey]] y los renombramos

![[petrick-example.png]]

- Para cada minterm tomamos la suma (booleana) de los implicantes que lo cubren.
- Multiplicamos todos los términos obtenidos en el paso anterior, de forma que todos los minterm estén cubiertos.

![[petrick-example2.png]]

- Desarrollamos el producto hasta obtener una expresión que no se pueda reducir más. Entonces cogemos un sumando, el que menor número de términos tenga y a ser posible que los minterm que intervengan tengan el mínimo número de literales.

![[petrick-example3.png]]
![[petrick-example4.png]]

- Ahora cogemos los implicantes de $B +  C + D$ (por ejemplo) y ya tendríamos la forma reducida.