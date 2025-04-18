- #r | #string | #datascience 

# Funciones
- **paste** -> Se utiliza para concatenar o pegar varios elementos de texto. Ej: `paste("Hola", "Mundo", sep=" ")`. 
	- `collapse` se usa para unir todos los elementos en un único string después de aplicar `sep` (si también lo especificas).
	- `sep` se utiliza para definir el separador entre cada elemento dentro de una misma llamada a `paste()`. Funciona en **cada par de elementos** que estás concatenando, es decir, cuando unes múltiples cadenas que se corresponden elemento por elemento.
- **substr** -> Devuelve una subcadena de la cadena original, se utiliza especificando el índice primero y último.
- **strsplit** -> Divide la cadenas o cadenas que se le pasen por parámetro por la subcadena especificada en el parámetro *split*. Se pueden especificar múltiples *splits* para cada cadena.
- **str_remove** -> Borra del string aquello especificado en el regex.

# Equivalente en tidyverse
| **R Base**            | **Tidyverse**                    | **Descripción**                              |
| --------------------- | -------------------------------- | -------------------------------------------- |
| `paste`               | `str_c`                          | Combina (concatena) cadenas de texto         |
| `substr`              | `str_sub`                        | Extrae o reemplaza subcadenas                |
| `nchar`               | `str_length`                     | Obtiene la longitud de una cadena            |
| `tolower`, `toupper`  | `str_to_lower`, `str_to_upper`   | Convierte a minúsculas o mayúsculas          |
| `grep`, `grepl`       | `str_detect`                     | Verifica la presencia de un patrón           |
| `gsub`, `sub`         | `str_replace`, `str_replace_all` | Reemplaza patrones en cadenas                |
| `strsplit`            | `str_split`                      | Divide cadenas según un patrón               |
| `regexpr`, `gregexpr` | `str_locate`, `str_locate_all`   | Encuentra posiciones de patrones             |
| `trimws`              | `str_trim`                       | Elimina espacios en blanco al inicio o final |
| `str_extract`         | `str_extract`, `str_extract_all` | Extrae coincidencias de un patrón            |
| `sprintf`             | `str_glue`                       | Formatea cadenas de texto con variables      |
| `str_dup`             | `str_dup`                        | Duplica cadenas el número de veces indicado  |
| `str_count`           | `str_count`                      | Cuenta ocurrencias de un patrón              |
