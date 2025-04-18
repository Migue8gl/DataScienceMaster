- #googlecloud | #looker | #dataanalyst 

- **Looker** es una herramienta de #businessintelligence que te permite visualizar grandes volúmenes de datos. Es la herramienta de Google Cloud Platform que se conecta con nuestras bases de datos y que permite visualizar estos datos en dashboards. 
- Dentro de Looker tendremos acceso al explorador, que es una interfaz para construir reportes sobre los datos. Dentro del explorador tendremos acceso a las vistas, que son campos que recojen ciertos datos (tablas de BD).

## Dimensiones y métricas
- Para crear reportes podemos acceder a las **dimensiones** (columnas o atributos) de nuestros campos para filtrar y aplicar métricas (conteo, media, etc.). Estas métricas son cálculos que se aplican sobre múltiples filas de nuestros datos.

![[looker-data-report-example.png]]
![[looker-visualization-example.png]]

- Podemos además añadir filtros para centrarnos en datos específicos basándonos en ciertas características. Por ejemplo, en la imagen anterior tenemos las ciudades con más usuarios (concretamente las top 10). Podemos añadir un filtro que nos responda esa misma pregunta, pero solo para el estado de California.
- Otra técnica interesante es el **pivoting** (pivotaje). Los pivotes nos permiten transformar una dimensión concreta en múltiples columnas, lo cuál crea una matriz de nuestros datos.

![[looker-pivoting-before.png]]
![[looker-pivoting-after.png]]

## Dashboards
- Si encontramos información interesante en nuestros datos, esta puede ser guardado en un **dashboard** o un **look**.
- Un look es un reporte o visualización individual, mientras que un dashboard es un conjunto de looks agrupados en una sola página.
- Looker y los datos, visualizaciones del mismo pueden ser conectados a través del conector de Looker Studio a esta herramienta, de forma que se pueden generar reportes de manera sencilla.

## Table calculations
- Estas son una forma de crear nuevas métricas *on-the-fly*, es decir, se crean en el momento para ver si deberían añadirse como métrica a largo plazo o para responder una pregunta aislada que necesite de esa métrica.
- Esta se definen con fórmulas personalizadas.
- Existen cuatro tipos -> **string**, **numérica**, **lógica**, **fecha (datetime)**.
- Ejemplo de tipo cálculo sobre tabla de tipo **string**:

![[table-calculation-looker.png|400]]

### Funciones offset
- Las funciones offset son un subconjunto de funciones de **cálculos sobre tablas**. Nos permiten de forma programática referenciar valores de otras filas o columnas para calcular nuevos valores.
- Hay tres tipos principalmente:
	- **offset()** -> se utiliza cuando se quiere referenciar un valor por encima o debajo de una fila. Ejemplo:

	 ![[offset-looker-function.png|300]]

	- **pivot_offset()** -> se utiliza cuando se desea referenciar un valor a la derecha o a la izquierda de una columna. Ejemplo:

	 ![[pivot-offset-looker-function.png|300]]

	- **offset_list()** -> se utiliza para combinar varios valores de múltiples filas en una sola celda dada una operación. Ejemplo:
	
	 ![[offset_list-looker-function.png|300]]
	