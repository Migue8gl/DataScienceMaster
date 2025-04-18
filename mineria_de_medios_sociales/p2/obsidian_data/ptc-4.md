- #tema4 | #python | #ptc

## OOP 
- La programación orientada a objetos (OOP) es un paradigma de programación que se basa en el concepto de "objetos". Estos objetos son instancias de clases, que son prototipos para la creación de objetos. Cada objeto puede contener datos, en forma de atributos, y código, en forma de métodos.
- Una clase define un tipo de datos junto a las operaciones y estado de este.

### Elementos:
- `__init__` -> constructor de una clase.
- `__del__` -> se llama antes de destruir el objeto.
- `__repr__` -> representación del objeto en string.
- `__str__` -> objeto transformado a string.
- Se usa la palabra `self` para acceder al objeto.
- Se pueden ver todas las capacidades de una clase usando `dir(objeto)`.

- Todas las clases de Python heredan de la clase *Object*.
- Al crear un objeto: `x = Coche()`, se devuelve una referencia.

### Herencia
- La clase padre se indica entre paréntesis:
```python
# Definición de la clase base (superclase)
class Animal:
    def __init__(self, nombre):
        self.nombre = nombre

    def hacer_sonido(self):
        pass

# Definición de una subclase que hereda de Animal
class Perro(Animal):
    def __init__(self, nombre, raza):
        # Llamamos al constructor de la superclase para inicializar el atributo 'nombre'
        super().__init__(nombre)
        self.raza = raza

    def hacer_sonido(self):
        return "Woof!"

# Definición de otra subclase que hereda de Animal
class Gato(Animal):
    def __init__(self, nombre, color):
        # Llamamos al constructor de la superclase para inicializar el atributo 'nombre'
        super().__init__(nombre)
        self.color = color

    def hacer_sonido(self):
        return "Meow!"
```

### Copia profunda
- `import copy` y usar método -> `box2 = copy.deepcopy(box)`.

### Método intrínsecos
- `__dict__` -> permite conocer atributos de una clase y sus valores en un tipo de dato diccionario.
- `__call__` -> permite usar un objeto como una función. Ej:
	```python
	class Factorial:
	    def __init__(self):
	        self.cache = {}
	
	    def __call__(self, n):
	        if n not in self.cache:
	            if n == 0:
	                self.cache[n] = 1
	            else:
	                self.cache[n] = n * self.__call__(n - 1)
	        return self.cache[n]

	```

### Sobrecarga de operadores
- **Aritméticos:**
  - `__add__`: `+` (suma)
  - `__sub__`: `-` (resta)
  - `__mul__`: `*` (multiplicación)
  - `__truediv__`: `/` (división verdadera)
  - `__floordiv__`: `//` (división entera)
  - `__mod__`: `%` (módulo)
  - `__pow__`: `**` (potencia)

- **Comparación:**
  - `__eq__`: `==` (igual a)
  - `__ne__`: `!=` (no igual a)
  - `__lt__`: `<` (menor que)
  - `__le__`: `<=` (menor o igual a)
  - `__gt__`: `>` (mayor que)
  - `__ge__`: `>=` (mayor o igual a)

- **Bitwise:**
  - `__and__`: `&` (AND bitwise)
  - `__or__`: `|` (OR bitwise)
  - `__xor__`: `^` (XOR bitwise)
  - `__lshift__`: `<<` (desplazamiento a la izquierda)
  - `__rshift__`: `>>` (desplazamiento a la derecha)
  - `__invert__`: `~` (inversión bitwise)

- **Asignación:**
  - `__iadd__`: `+=` (suma en asignación)
  - `__isub__`: `-=` (resta en asignación)
  - `__imul__`: `*=` (multiplicación en asignación)
  - `__itruediv__`: `/=` (división verdadera en asignación)
  - `__ifloordiv__`: `//=` (división entera en asignación)
  - `__imod__`: `%=` (módulo en asignación)
  - `__ipow__`: `**=` (potencia en asignación)
  - y otros similares...

- **Representación de Cadena:**
  - `__str__`: `str(obj)` (cadena legible para humanos)
  - `__repr__`: `repr(obj)` (cadena no ambigua que representa el objeto)

- Ej:
	```python
	class Punto:
	    def __init__(self, x, y):
	        self.x = x
	        self.y = y
	
	    def __add__(self, otro_punto):
	        if isinstance(otro_punto, Punto):
	            nuevo_x = self.x + otro_punto.x
	            nuevo_y = self.y + otro_punto.y
	            return Punto(nuevo_x, nuevo_y)
	        else:
	            raise ValueError("Se espera un objeto de tipo Punto para la suma")
	
	    def __str__(self):
	        return f"({self.x}, {self.y})"
	```

### Métodos de clase y estáticos
- Los métodos de clase se utilizan cuando necesitas interactuar con la propia clase, mientras que los métodos estáticos se usan cuando la operación no necesita acceder a los atributos de instancia ni de clase
- `@classmethod` -> se utiliza antes del método y por convención se le debe pasar el parámetro `cls` que representa la propia clase.
- `@staticmethod` -> se utiliza antes del método, no recibe parámetro especial. Cuando se requiere hacer una operación común a todas las instancias de la clase.

### Clases abstractas
- Importar -> `from abc import ABCMeta, abstractmethod, abstractproperty`
- Simplemente se utiliza `@abstractmethod` o `@abstractproperty` en los métodos o propiedades y se utiliza `pass` para no implementarlos.

## Pandas
- Es un módulo de Python que permite trabajar con ficheros varios como *JSON*, *CSV*, *Excel*, etc.
- Permite utilizar grandes volúmenes de datos usando la estructura **Dataframe**.
- Técnicas de análisis de datos.
- Para importar -> `import pandas as pd`

### Dataframe
- Representa una tabla, de forma similar a una hoja de cálculo.
- Contiene una colección ordenada de columnas que pueden tener distintos tipos de valores (numérico, string, lógico, etc.).
- Tiene índices por filas y por columnas.
- Pandas se usa para todo tipos de datos, mientras que Numpy está limitado a datos numéricos. Pandas es más eficiente cuando se usan más de $500.000$ filas, Numpy lo supera para menos de $50.000$ filas.
- Para crear un dataframe:
	- `frame = pd.DataFrame(data)`. Siendo `data` normalmente un diccionario.
	- Se puede ordenar un dataframe usando:
		- `frame.sort_values(by=['key',...], ascending=[True,...])`
		- Se pueden ordenar las columnas especificando el orden al crear el *DataFrame* -> `pd.DataFrame(data, columns=[...]`		
- Para ver cuales son las claves o columnas -> `frame.columns`
- Para ver una clave o columna entera -> `frame.<nombre de la columna>`. Aunque también se puede acceder con el operador `[]`.
 - Modificar valores de una columna con `np.arange`:
	 - `frame['año'] = np.arange(2010, 2023, 1)`
- Asignar una serie/lista a una columna:
	- `val = pd.Series([...], index=[0, 3, 8,...])`
	- `frame['deuda'] = val`
	- Cambia en la columna aquellos valores de los índices indicados por los nuevos valores de la serie.
- Si no existe la columna al intentar acceder a ella, se añade.
- Para borrar columnas -> `del frame['deuda']`
- Si usamos un diccionario de diccionarios se creará un **DataFrame** cuyos índices para filas serán las *keys* de los diccionarios dentro del diccionario padre.
- Se puede hacer la traspuesta del **DataFrame** con -> `frame.T`
- Se pueden asignar índices explícitos usando `index` en la creación del **DataFrame**.
- Se puede hacer uso de creación de índices para mejorar la búsqueda de una columna:
	- `frame.set_index(['región'], inplace=True)`
	- Búsqueda -> `frame.loc['Andalucía']`
- Se le pueden poner nombre a las columnas y filas:
	- `frame.index.name = 'dnfkn'`
	- `frame.columns.name = 'dnfkn'`
- Eliminar y rellenar valores faltantes:
	- `frame.dropna()`
	- `frame.fillna(value)`
	- `frame.interpolate()` -> Por defecto se usa interpolación lineal.
- Aplicar funciones a columnas concretas:
	- `df[['Matematicas', 'Ciencias']] = df[['Matematicas','Ciencias']].applymap(redondear_a_entero)`
	- `applymap` se dejará de usar a favor de `map`.
- Iterar:
	- `for index, row in df.iterrows():`
	- `for column in df:`
- Búsquedas:
	- `df[df['Ventas'] > 1000]`
	- `df[(df['Ventas'] > 1000) & (df['Ventas] < 2000)]`
- Se pueden añadir filas a un dataframe, siempre que el otro tenga las mismas columnas:
	- `df.append(df2)`

## Decoradores y wrappers
- Los decoradores o *decorators* son un patrón de diseño que permite extender el comportamiento de funciones o métodos sin cambiar el código original.
- Un decorador NO es un **wrapper** cuando retorna la propia función que se le pasa:
	```python
		def non_wrapping_decorator(func):
		    return func # function out is the same as the function in
		
		@non_wrapping_decorator
		def function_not_wrapped():
		    pass
	```
- En cambio, si retorna otra cosa, es un wrapper:
	```python
		def wrapping_decorator(func):
			def wrapper():
			    return func()
			return wrapper 
	```
	
## Scikit-learn
- Importar módulo -> `import sklearn`
- Podemos importar **datasets** preconstruidos como los famosos *iris* y *digits*:
	- `from sklearn import datasets`
	- `iris  = datasets.load_iris()`
- Estos datasets son diccionarios de Python que contienen datos y metadatos.
- También podemos cargar datasets en formato **DataFrame** con el módulo **Seaborn**:
	- `import seaborn as sns`
	- `iris = sns.load_dataset('iris')`
	- Para ver las primeras filas del DataFrame de pandas -> `iris.head()`
	- Ejemplo de visualización con seaborn:
		```python 
		import seaborn as sns 
		
		sns.set(style="ticks", color_codes=True) 
		sns.pairplot(iris, hue='species', height=1.5) 
		
		import matplotlib.pyplot as plt
		plt.show()
		```
	-  Este tipo de gráfico muestra relaciones entre variables para una variable categórica dada, que es el parámetro `hue`.

### Aprendizaje
- Un estimador utiliza los métodos:
	- `fit(x,y)` -> Realiza el aprendizaje por parte del algoritmo estimador para ajustar una serie de pesos, de parámetros se le da un conjunto de datos y etiquetas asociadas a cada dato.
	- `predict(t)` -> Dado un conjunto de datos estima las etiquetas de estos a partir de los pesos aprendidos previamente.
- En el estimador **LinearRegression**, el parámetro `fit_intercept` debe ponerse a *True* si los datos no están centrados en el eje de coordenadas, o por defecto se usara un [[essential math for data science#Linear Regression|intercepto]] de $(0,0)$ -> [mirar](https://stackoverflow.com/questions/46779605/in-the-linearregression-method-in-sklearn-what-exactly-is-the-fit-intercept-par)