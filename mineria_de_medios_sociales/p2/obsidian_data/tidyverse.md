- #tidyverse | #r 

# Concepto
- **Tidyverse** es un conjunto de paquetes diseñados para facilitad el análisis de datos y la visualización.
- La filosofía "tidy" se centra en la organización de datos de tal manera que:
	- Cada variable se encuentra en una columna.
	- Cada observación se encuentra en una fila.
	- Cada tipo de unidad observacional forma un tabla.
- Esta estructura hace que el análisis de datos y la visualización sean más directos y claros.

# Características
## Operador de pineline
- El operador `%>%` sirve para concatenar varias operaciones de forma secuencial sin recurrir a parentesis anidados. Ej:
	- Sin pipeline: `round(mean(sqrt(x)))`.
	- Con pipeline: `x %>% sqrt() %>% mean() %>% round()`.
## Manipulación de datos
- **filter** -> Selecciona filas de un *data frame*. Ej: `starwars %>% filter(species == "Droid")`.
- **slice** -> Selecciona filas específicas basandose en los índices dados. Ej: `starwars %>% slice(c(1,4,6))`.
- **top_n** -> Permite seleccionar las top $n$ filas que tienen los valores más altos en una columna determinada. Si $n$ se especifica como negativo, entonces buscando los valores más bajos.
- **select** -> Permite seleccionar columnas del *data frame*. Si a la variable o variables seleccionadas le añades el signo negativo, significará que se seleccionan todas excepto esas especificadas.
- **arrange** -> Permite ordenar las filas de una *data frame* por la variable especificada. Si se usas `desc` el orden será descendiente. Ej: `starwars %>% arrange(height, desc(birth_year))`.
- **rename** -> Renombra columnas.
- **mutate** -> Crea o actualiza columnas de un *data frame*. Ej: `starwars %>% mutate(height = height * 0.5, new_name = paste("Sr", name))`.
- **summarize** -> Permite resumir filas en un *data frame*. Ej: `starwars %>% summarize(mean = mean(height, na.rm = TRUE), number_of_rows = n())`.
- **across** -> Permite aplicar una operación a varias columnas de manera simultánea.
- **pull** -> Extrae una columna como un vector.