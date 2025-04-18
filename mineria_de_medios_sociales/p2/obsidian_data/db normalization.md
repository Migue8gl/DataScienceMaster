- #db | #sql | #normalization | #1F2F3F

## Introducción
- La normalización hace que la estructura de la base de datos no pueda expresar redundancias.
- Más fáciles de entender, de mejorar y de extender. Además proteje de anomalías en las operaciones de inserción, actualización y borrado.
- Cada nivel de normalización es una evaluación de seguridad de la DB.

## Primera forma normal (1F)
- El orden de las filas no puede transmitir información. Cada fila es independiente y puede reordenarse sin perder significado.
- Mezclar tipos de datos en la misma columna no se permite.
- Se debe tener una clave primaria que identifique a cada fila.
- Los grupos repetitivos no se permiten. No debe haber columnas que contengan listas o conjuntos de valores.

## Segunda forma normal (2F)
- Debe estar en 1F.
- Cada atributo que no sea una clave debe depender completamente de la clave primaria.

![[db normalization-2F-example.png|600]]

## Tercera forma normal (3F)
- Debe estar en 2F.
- No se debe permitir dependencias entre atributos que no sean clave.

![[db normalization-3f-normalization.png]]