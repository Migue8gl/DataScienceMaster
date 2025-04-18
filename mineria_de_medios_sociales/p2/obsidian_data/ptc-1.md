- #tema1 | #python | #ptc

## Asignaciones
- Las asignaciones en Python crean referencias, no copias, de forma que $x=5$ indica que donde se guarde el valor $5$ en memoria puede ser referenciado usando la etiqueta $x$.
- Con la función *id* podemos obtener la dirección en memoria:
	- `x = 5`
	- `id(x) = 505910929`
- Podemos intercambiar variables y asignar valores e la siguientes manera:
	- `x,y = 2,3`
	- `x = y = c = 2` (Tienen el mismo id).
	- `x,y = y,x` (Swap de valor).

## Convención 
- Se suele usar *snake_case* para todo excepto:
	- Clases: *CamelCase*.
	- Atributos: Se declaran el método \_\_init\_\_.
	- Constantes: mayúsculas.

## Funciones
- Se utiliza `def`.
- Se pueden devolver varios valores. 
- Se documenta:```
	def func(n):
	"""
	Esta función hace x.
	"""
		-- Resto de la función --```
- Si se requiere la doc de una función -> `func.__doc__`.
- Se pueden usar alias para una función:```
	f = func
	x = f(n)```
- Se pueden usar valores por defecto en las funciones.
- Número de parámetros variable en una función -> `def func(*datos)`.
- Parámetros con nombre en una función -> `def func(**datos)`.

### Funciones lambda
- Las funciones **lambda** son funciones anónimas y se crean de la siguiente forma: `<func> = lambda <params> : <return values>`. Ej:
	- `suma = lambda x,y : x+y`

## Datos básicos
- Entero -> int.
- Lógico -> bool (True y False (la primera en mayúscula)).
- Real -> float (si su representación en base 2 no es exacta se pierde precisión). Ej:
	- `0.1 + 0.1 + 0.1 = 0.30000000000000004`
- Complejo -> complex.
- Cadena de caractéres -> str.

### Módulo math
- `import math`
- `exp, log, sin, cos, ...`
- Se usan con datos *float*.

### String
- Son inmutables.
- Se pueden usar comillas simples o dobles.
- Operadores `[], +`.
- Si es multilínea se ha de usar triple entrecomillado de dobles comillas.
- `len, lower, lstrip, replace, split, swapcase, upper, count, find, join, partition, str`.
- Para usar `\` como caracter hay que usar:
	- `\\'palabra'` o `r'\palabra'` 
- Ejemplos:
	- `x = Hola mundo`
	- `x[0:-1]` -> `Hola mund`
	- `x[::-1]` -> `odnum aloH`
	- `x[-1:2:-1]` -> `odnum a`

## Duck Typing (datos dinámicos)
- El tipo de cada variable se determina de forma dinámica.
- Con `type(x)` o `isinstance(x, type)` podemos saber el tipo de cada variable. El primero devuelve el tipo y el segundo es booleano.
- No se pueden operar datos de distinto tipo a no ser que se haga *casting* de manera explícita (tipo dinámico fuerte).

## E/S y estructuras de control
- `x = input('Introduce un dato: ')`
- `x = int(input('Introduce un dato entero: '))`
- `print('El cuadrado de', x, 'es', x*x)`
- `print('El cuadrado de %d es %d, x*' % (x, x*x))`
- `print('El cuadrado de {} es {}'.format(x, x*x))`
- Para evitar el separador por defecto (espacio en blanco) y el final de línea por defecto (salto de línea) se usan:
	- `print(end=, sep=)`

### If else
- Función lambda con *if-else*:
	- `fact = lambda n : 1 if n == 1 or n == 0 else n * fact(n-1)`

### Bucles
- Se usa `zip()` para iterar sobre los elementos que se pasen a la vez. Si se pasan dos listas, devuelve tuplas de dos en dos.

## Datos complejos
- Tupla -> tuple.
- Lista -> list.
- Diccionario -> dict.
- Conjunto -> set.
- Array -> array.

### Datos Secuencias
- Índice positivo (por la izquierda) empezando por $0$, índice negativo (por la derecha) empezando por $-1$.
- Para trocear una tupla, lista o string:
	- `[i:j]` -> Empieza en `i` y termina **antes** de `j`.
	- `[:i]` -> Desde el primero hasta antes de `i`.
	- `[i:]` -> Desde `i` hasta el final.
	- `[:]` -> Copia de la secuencia, es decir, si asignas una variable a otra, ambas comparten referencia y por tanto cambiar una modifica la otra. Si se iguala una variabla a la copia de secuencia, entonces ya no comparten memoria.
- **Inmutables** -> tupla (*tuple*), cadena de caracteres (*str*).
- **Mutables** -> lista (*list*).
- La inmutabilidad permite mayor velocidad.
- El operador `in` comprueba si un valor se encuentra en un contenedor o cadena (en una secuencia).
- Se pueden concatenar listas (producen una nueva referencia) o repetir la secuencia original (produce copia) con -> `+, *`.
- (**Listas**) Insertado -> `lista.append(x)`.
- (**Listas**) Insertado en posición `i` -> `listas.insert(i, x)`.
- (**Listas**) Opera en lista y extiende esta con otra lista, no hace copia. -> `lista.extend(lista2)`.
- (**Listas**) Devuelve el índice del elemento (primera aparición). -> `lista.index(a)`.
- (**Listas**) Cuenta apariciones del elemento. -> `lista.count(a)`.
- (**Listas**) Borra el elemento (primera aparición). -> `lista.remove(a)`.
- (**Listas**) Invierte la lista -> `lista.reverse()`.
- (**Listas**) Ordena la lista -> `lista.sort()`. Podemos pasarle como queremos que ordene. Ej:
	- `lista = ['aaa', 'aaaaaa', 'a']`
	- `lista.sort(key = lambda n : len(n))`
- La coma es el operador de creación de la tupla.
- Las [[ptc-1#^1865e4|funciones que devuelven varios valores]] devuelven una tupla que contiene esos valores.
- La función `enumerate(secuencia)` devuelve un par índice, elemento.
- La función `sorted(secuencia)` devuelve una lista ordenada.
- La función `reversed(secuencia)` devuelve una lista con los elementos en orden inverso.
- Otras operaciones sobre listas -> `max(), min(), sum(), len()`.
- Operaciones sobre strings + list -> `lista.strip(), lista.split()`.

### Datos asociativos
- Son conjuntos ordenados y pueden ser de distinto tipo (set y dict). Son mutables.
- Los conjuntos deben ser hashables (que se pueda aplicar un hash) -> datos **inmutables**.
- Las claves de los diccionarios *deben* ser inmutables.
- Se puede usar `zip` para crear un diccionario:
	- `dict(zip(claves, valores))`
- Para borrar un elemento del diccionario se usa `del`.
- Para obtener las valores -> `items()`.
- Para obtener las claves -> `keys()`.
- Cuando se usa `in` se comprueba si *existe* la clave.
- Operadores sobre *set*:
	- `conj.add(), conj.remove(), conj.copy(), conj.clear()`.
	- Incluye operadores de unión, intersección y diferencia (`union, intersection, difference`). 
	- Unión es todo elemento de *A* que esté en *B*.
	- Intersección todo elemento que esté en *A* y en *B*.
	- Diferencia es todo elemento de *A* que no esté en *B*.
	- Diferencia simétrica es todo elemento que está en *A* o en *B*, pero no en los dos a la vez.

## Comprensión de listas
- Se puede crear una lista con todos los valores utilizando una única instrucción. Es muchísimo más eficiente que con bucles for:
	- `nueva_lista = [expresion for elemento in iterable if condicion]`

## Fichero
- Para abrir un fichero:
	- `open(<nombre>, <modo>`. Donde el modo puede ser `r,w,a`.
	- `f = open('datos.txt', 'r')
- Para cerrar un fichero -> `f.close()`.
- Para comprobar si existe un fichero:
	- `os.path.isfile(<nombre>)`
- Para comprobar permisos de un fichero:
	- `os.stat(<nombre>)`
- Para leer una línea completa de un fichero:
	- `f.readline()`
	- `f.readlines()` lee el fichero por líneas completo (en una lista) (<u>incluye saltos de línea</u>).
- Iterar sobre líneas:
	- `for linea in f`
- También se puede hacer en un bloque (fuera del bloque el archivo se cierra):
	- `with open(<nombre>, <modo>) as f`
- Podemos escribir una secuencia con varias líneas o en solo una:
	- `f.writelines(secuencia)` o `f.write(cadena)`

### E/S estándar
- Se debe importar `sys`.
- El siguiente código agrupa la entrada y salida estándar:
```
import sys
linea = sys.stdin.readline()

while linea:
	sys.stout.write(linea)
	linea = sys.stdin.readline()
```

### Ejemplo lecturas csv
```
with open('datos.csv', 'r') as f:
	datos = f.readline().split(',')
```

```
import csv
with open('datos.csv', 'r') as f:
	data = list(tuple(rec) for rec in csv.reader(f, delimiter = ','))
```

### Guardar estructuras de datos
```
lista = ['a', 'b']
a = [[1.2, lista], 'text']
f = open('tmp.dat', 'w')
f.write(str(a))
f.close()
```

### Leer estructuras de datos
```
f = open('tmp.dat', 'r')
datos = eval(f.readline())
f.close()
```

## Trabajar con datos en disco (DB)
- Permite manejar datos sin cargarlos.
- Hay que importar `shelve`.
- Para abrir la base de datos:
	- `database = shelve.open('db.bin')`
- Para insertar datos y consultarlos es igual que un diccionario.
- Se puede usar el operador `in` para ver si alguna clave existe en la base de datos.

## Programación funcional
- Python utilza funciones de orden superior, esto es, funciones que utilizan otras funciones como parámetros. Ejemplo de estas son:
	- *map*, *reduce*, *filter*.

### Map
- Uso -> `map(<function>, <iterable>, ...)`
- Se aplica la función a cada uno de los elementos del objeto y crea una secuencia con los resultados. Ej:
	- `list(map(lambda x : x**2, range(10, 20)))` -> Devuelve una lista con los números $10$ a $20$ al cuadrado. 
- Ha de transformarse a lista pues `map` devuelve un iterador.
- Se pueden pasar tantos objetos iterables como se quiera.
- Se puede hacer uso de un map más eficiente utilizando *multiprocessing.
	- `import multiprocessing`
	- `pool = multiprocessing.Pool()`
	- `pool.map(....)`
	- `pool.close()`

### Filter
- Uso -> `filter(<function>, <iterable>)`
- Procesa cada elemento del objeto iterable aplicándole la función.
- Si la función devuelve True para ese elemento, entonces filter lo devuelve.
- Es necesario hacer un cambio a tipo list.

### Reduce
- Uso -> `map(<function>, <iterable>[,initializer])`
- Se aplica a cada elemento del objeto iterable junto con el resultado hasta el momento, acumulando resultados.
- La función debe tener dos argumentos.
- Si existe inicializador, este se utiliza como primer argumento.
- Hay que importar `from functools import reduce`.

## Manejo de excepciones
- Las excepciones son eventos qeue pueden modificar el flujo del programa.
- Se lanzan automáticamente con cada error.
- `try/except` -> capturar y recuperar la excepción.
- `try/finally` -> realizan acciones si la excepción de produce o no.
- `raise` -> genera una excepción manualmente.
- `assert` -> genera una excepción de manera condicional.

## Generadores
- Permiten iterar sobre un conjunto de elementos una única vez.
- No se guardan los números en memoria, solo se producen cuando sean necesarios.
- Se utilizan usando la palabra clave `yield`.
- Suponen un incremento de rendimiento en cuanto a memoria y velocidad.
- Ejemplo:
	```
	import itertools
	def fibonacci_generator():
	a, b = 0, 1
	while True:
		yield a
		a, b = b, a+b
	
	print(list(itertools.islice(fibonacci_generator(), 10)))
	```
- Se pueden crear incluso secuencias infinitas.

## Anotaciones
- Las anotaciones aportan una manera de añadir metadatos con el objetivo principal de documentar el código.
- Puede accederse mediante `func.__annotations__`. Ej:
	```
	def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers and returns the result.

    :param a: An integer, the first number.
    :param b: An integer, the second number.
    :return: An integer, the sum of a and b.
    """
    return a + b

	print(add_numbers.__annotations__)
	# Output: {'a': <class 'int'>, 'b': <class 'int'>, 'return': <class 'int'>}
	```

## Cálculo simbólico
- Este se realiza mediante el módulo de **Sympy**.
- Podemos utilizar infinitud de funciones de ámbito matemático y mostrarlas de forma simplificada o como ecuación sin evaluar. Por ejemplo, con las fracciones, usando `Rational`, en vez de evaluar $1/2$ como $0.5$, muestra la fracción.
- Si queremos evaluar cualquier expresión, vale con usar `evalf(x)` donde `x` es la precisión con la que se muestra el valor.
- Se pueden expandir expresiones con `expand(expr)`, o simplificarlas con `simplify(expr)`. Ej:
	- `sym.sin(x) / sym.cos(x)` Con simplificación -> `tan(x)`.
- Se pueden hacer derivadas, límites, series de taylor, integrales, resolver ecuaciones, etc.
