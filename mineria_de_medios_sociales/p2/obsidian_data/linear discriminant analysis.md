- #lda | #ml 

# Introducción
- El algoritmo **LDA** realiza tareas de clasificación, como por ejemplo la [[logistic regression|Regresión Lógística]].
- El algoritmo **LDA** realiza lo mismo que [[pca|PCA]], pero maximizando la separabilidad de las categorías en vez de maximizar la varianza (minimizar pérdida de información).

# Funcionamiento
- El algoritmo **LDA** crea un nuevo eje en el espacio y proyecta los datos a ese nuevo eje.
![[linear discriminant analysis-lda-example1.png|500]]
- Luego maximiza la función que representa la diferencia de medias, normalizadas por una medida de variabilidad dentro de la clase.
![[linear discriminant analysis-example2.png|500]]
- Se extraen las que se conocen como variables *latentes*, llamadas **discriminantes**, formadas por combinaciones lineales de las variables independientes. Los coeficientes en esas combinaciones lineales son los **coeficientes discriminantes**.
- Los datos son asignados a las clases por los discriminantes, no por las variables originales.
- Los coeficientes se extraen creado un espacio de baja dimensionalidad donde se maximice la varianza entre clases y se minimice la varianza intra-clase.

# Asunciones
- Asume que cada variable independiente tiene una distribución normal para cada clase.
- Cada clase tiene matrices de varianzas-covarianzas idénticas.
- Las observaciones vienen de una muestra aleatoria.