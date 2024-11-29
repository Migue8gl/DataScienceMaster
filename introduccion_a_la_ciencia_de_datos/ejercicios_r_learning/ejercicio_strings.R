library(tidyverse)
"Crea un vector de cadenas de caracteres con tu nombre y apellidos
(por ejemplo, [\"Rocío\", \"Romero\", \"Zaliz\"]). A partir de el crea una
nueva cadena de caracteres con la inicial de tu nombre, un punto y tus apellidos
(por ejemplo, \"R. Romero Zaliz\")"

my_name <- c("Miguel", "García", "López")
paste(substr(my_name[1], 1, 1), ". ", my_name[2], " ", my_name[3], sep = "")

"Dado un vector de cadenas de caracteres que representan fechas (por ejemplo,
[\"2005-11-28\", \"2015-10-18\", \"2000-01-01\"], utilizando el formato
AÑO-MES-DÍA), mostrar sólo las correspondientes a los meses impares."

dates <- c("2005-11-28", "2015-10-18", "2000-01-01")
odd_months <- c("01", "03", "05", "07", "09", "11")
months <- matrix(unlist(strsplit(dates, "-")), nrow = length(dates), byrow = TRUE)[, 2]
dates[months %in% odd_months]

"Dada una cadena de caracteres con varias palabras (por ejemplo, \"Esta es una
frase, pero no cualquier frase.\") crea un vector con cada una de las palabras
de la cadena (por ejemplo, [\"Esta\", \"es\", \"una\", \"frase\", \"pero\",
\"no\", \"cualquier\", \"frase\"]). Tenga en cuenta todos los caracteres de
puntuación posibles."

strsplit(gsub("[^a-zA-Z0-9 ]", "", "Esta es una frase, pero no cualquier frase."),
         split = " ")[[1]]

"Busca en un vector de cadenas de caracteres aquellas que incluyan sólo vocales
\"a\" y/o \"e\" o ninguna (comprueba mayúsculas y minúsculas, considera á, é, Á
y É como otros caracteres y no los que buscas)."

cadenas <- c("a",
             "e",
             "aa",
             "ae",
             "aeiou",
             "bcd",
             "xyz",
             "AAA",
             "eee",
             "ÁE",
             "EEE",
             "ab",
             "abc")

cadenas[grepl("^[aeAE]*$", cadenas)]

"Dados tres vectores numéricos que representan días, meses y años, crea un
vector nuevo con fechas (sólo si son válidas, si la fecha es inválida ignorarla)
(Sugerencia: investigue la función as.Date)."

days <- c(01, 2, 4, "8")
months <- c(12, 1, 23, "02")
years <- c("2011", 1970, 2001, "-1")

as.Date(paste(days, months, years, sep = "-"), format = '%d-%m-%Y')

# ------------------------------- 2da parte ------------------------------------

"Crea un vector de cadenas de caracteres con tu nombre y apellidos
(por ejemplo, [\"Rocío\", \"Romero\", \"Zaliz\"]). A partir de el crea una
nueva cadena de caracteres con la inicial de tu nombre, un punto y tus apellidos
(por ejemplo, \"R. Romero Zaliz\")"

my_name <- c("Miguel", "García", "López")
str_c(str_sub(my_name[1], 1, 1), ". ", my_name[2], " ", my_name[3], sep = "")

"Dado un vector de cadenas de caracteres que representan fechas (por ejemplo,
[\"2005-11-28\", \"2015-10-18\", \"2000-01-01\"], utilizando el formato
AÑO-MES-DÍA), mostrar sólo las correspondientes a los meses impares."

dates <- c("2005-11-28", "2015-10-18", "2000-01-01")
odd_months <- c("01", "03", "05", "07", "09", "11")
months <- matrix(unlist(strsplit(dates, "-")), nrow = length(dates), byrow = TRUE)[, 2]
dates[months %in% odd_months]

"Dada una cadena de caracteres con varias palabras (por ejemplo, \"Esta es una
frase, pero no cualquier frase.\") crea un vector con cada una de las palabras
de la cadena (por ejemplo, [\"Esta\", \"es\", \"una\", \"frase\", \"pero\",
\"no\", \"cualquier\", \"frase\"]). Tenga en cuenta todos los caracteres de
puntuación posibles."

str_split(str_replace_all("Esta es una frase, pero no cualquier frase.", "[^a-zA-Z0-9 ]", ""),
          " ")[[1]]

"Busca en un vector de cadenas de caracteres aquellas que incluyan sólo vocales
\"a\" y/o \"e\" o ninguna (comprueba mayúsculas y minúsculas, considera á, é, Á
y É como otros caracteres y no los que buscas)."

cadenas <- c("a",
             "e",
             "aa",
             "ae",
             "aeiou",
             "bcd",
             "xyz",
             "AAA",
             "eee",
             "ÁE",
             "EEE",
             "ab",
             "abc")

cadenas[str_detect(cadenas, "^[aeAE]*$")]

"Dados tres vectores numéricos que representan días, meses y años, crea un
vector nuevo con fechas (sólo si son válidas, si la fecha es inválida ignorarla)
(Sugerencia: investigue la función as.Date)."

days <- c(01, 2, 4, "8")
months <- c(12, 1, 23, "02")
years <- c("2011", 1970, 2001, "-1")

as.Date(str_c(days, months, years, sep = "-"), format = '%d-%m-%Y')

