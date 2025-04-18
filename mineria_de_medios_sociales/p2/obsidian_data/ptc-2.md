- #tema2 | #python | #ptc

## NumPy
- **Numpy** es un paquete de módulos que permite computación numérica de forma eficiente.

### Tipo de dato narray
- Vector n-dimensional rectangular de datos.
- Los datos pueden ser de cualquier tipo [[ptc-1#Datos básicos|simple]] y esta librería tiene algoritmos eficientes para estos tipos.
- Si `n` es el número de elementos de un narray, entonces `np.sin(x)` es equivalente a `[sin(x[i]) for i in range(n)]`. Lo mismo con `x+y`, siendos estos dos narrays, u otras operaciones de elemento a elemento.

![[narray-numpy.png]]

### Numpy dtypes
![[numpy-dtypes.png]]

### narray vs listas
- Los array de numpy deben tener filas de misma longitud, mientras que las listas no. Además las entradas deben ser del mismo tipo, lo que no ocurría con las listas.
- Las listas son muchísimo más ineficientes para cáculos algebráicos y básicos que los array de numpy.

### Funcionalidades varias
- Se pueden crear narrays a partir de una lista -> `np.array([1,2,3,4])`.
- Para comprobar la dimensión del vector -> `np.shape(a)` o `a.shape`. También pueden comprobarse las dimensiones con `a.ndim`.
- Para asignar una copia ha de hacerse -> `b = a.copy()`.
- Para vectores nulos o todo a cero -> `np.zeros((3,3))` (Crea una matriz $3\times 3$)
- `np.linspace(a, b, n)` -> genera `n` valores equiespaciados empezando en `a` y terminando en `b`.
- `np.arange(a, b, x, tipo)` -> genera un array con valores entre `a` y `b` (sin incluir `b`) de `x` en `x` valores.
- Se puede crear un array de numpy a partir de una función como en el siguiente ejemplo:
	```
	import numpy as np
	def mi_func(i, j):
	    return (i + 1) * (j + 4 - i)
	
	# Create a 3x6 array where a[i, j] = mi_func(i, j)
	a = np.fromfunction(mi_func, (3, 6))
	```
- También se pueden crear arrays aleatorios con -> `np.random.uniform(low, high, shape)`.
- La forma más eficiente de pasar un narray a una lista es con `a.tolist()`. De otra manera sería necesario usar `list` para cada dimensión.
- Se puede calcular el mínimo y máximo (en datos multidimensionales se usa `np.amin/np.amax`)

### Indexación indirecta
- Esto es útil cuando necesitamos acceder a elementos específicos en un patrón no regular, uso de filtros o máscaras, etc.
- Podemos obtener una lista de valores en unos índices concretos:
	- `y = a[[1, 2 ,-3]]`
	- `y = take(a, [1, 2, -3])`

- También se pueden obtener valores de un narray que cumplan una condición:
	- `a[a % 2 == 1]`
	- `a[a < 0]`

### Operaciones
- Como se ha dicho antes, las operaciones sobre un narray son elemento a elemento, de forma que se puede hacer -> `b = 2 * a -1`, siendo `a` un vector de numpy.
- Se pueden usar funciones estadísticas tales como `np.mean(), np.average(), np.std(), np.var()`, etc.
- Se puede limitar el valor máximo y mínimo de un vector con -> `np.clip()`.
- Se puede aplanar el vector a 1D con `np.flatten()`.
- Se puede calcular la traspuesta con `np.transpose()` o `a.T`.
- Suma/producto acumulado a lo largo de un eje -> `a.cumsum()/a.cumprod()`.

## Matplotlib
- Para importar la biblioteca de gráficos -> `import matplotlib.pyplot as plt`
- Diferentes tipos de plots:
	-  *Line plot*:
		`plt.plot(x, y, label='Line Plot')`
		`plt.xlabel('X-axis Label')`
		`plt.ylabel('Y-axis Label')`
		`plt.title('Line Plot Example')`
		`plt.legend()`
		`plt.show()`
	- *Scatter plot*:
		`plt.scatter(x, y, label='Scatter Plot', color='red', marker='o')`
		`plt.xlabel('X-axis Label')`
		`plt.ylabel('Y-axis Label')`
		`plt.title('Scatter Plot Example')`
		`plt.legend()`
		`plt.show()`
	- *Bar plot*:
		`plt.bar(x, y, label='Bar Plot', color='blue')`
		`plt.xlabel('X-axis Label')`
		`plt.ylabel('Y-axis Label')`
		`plt.title('Bar Plot Example')`
		`plt.legend()`
		`plt.show()`
	- *Histogram*:
		`plt.hist(data, bins=10, color='green', alpha=0.7)`
		`plt.xlabel('X-axis Label')`
		`plt.ylabel('Frequency')`
		`plt.title('Histogram Example')`
		`plt.show()`
	- *Box plot*:
		`plt.boxplot(data, vert=False, patch_artist=True)`
		`plt.xlabel('X-axis Label')`
		`plt.ylabel('Box Plot')`
		`plt.title('Box Plot Example')`
		`plt.show()`
	- *Pie chart*:
		`plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)`
		`plt.title('Pie Chart Example')`
		`plt.show()`
	- *Heatmap*:
		`plt.imshow(matrix, cmap='viridis', interpolation='nearest')`
		`plt.colorbar()`
		`plt.title('Heatmap Example')`
		`plt.show()`
	- *Contour map*:
		`contour_plot = plt.contour(X, Y, Z, cmap='viridis')`
		`plt.colorbar(contour_plot, label='Z values')`
		`plt.title('Contour Plot Example')`
		`plt.xlabel('X-axis')`
		`plt.ylabel('Y-axis')`
