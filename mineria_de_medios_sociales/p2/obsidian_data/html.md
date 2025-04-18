- #html 

## Table
- Elemento que contiene filas y columnas con información, debe usarse las etiquetas:
	- `<table>`
- Algunas etiquetas para las tablas son:
	- `<caption>` -> Título opcional.
	- `<colgroup` -> Define un grupo de columnas en una tabla.
	- `<thead>` -> Define un grupo de filas, opcional, que definan la cabeza de la tabla. Info semántica.
	- `<tbody>` -> Define un conjunto de filas de la tabla (cada fila es un `<tr>`). Info semántica.
	- `<tr>` -> Cada una de las filas (elementos de la tabla). Cada fila puede además llevar las siguientes etiquetas:
		- `<td>` -> Define un dato de una celda.
		- `<th>` -> Define una celda cabecera. Se debe definir un *scope*.
		- Podemos definir algunos atributos, dentro de estos tags, como por ejemplo **colspan** o **rowspan** para definir la extensión de cada celda. Un colspan de 5 sería una celda que abarca 5 columnas.