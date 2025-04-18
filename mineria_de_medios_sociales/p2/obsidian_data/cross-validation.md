- #cross-validation | #validacioncruzada | #k-fold

### K-Fold Cross-validation
- Dado un parámetro $k$ se divide la muestra de datos en $k$ grupos. Para lograrlo se siguen los siguientes pasos:
	- Se mezclan los datos de manera aleatoria.
	- Se dividen los datos en $k$ grupos.
	- Para cada grupo:
		- Se escoge si será test o train.
		- El resto de particiones se quedan como conjuntos de entrenamiento (train).
		- Se ajusta el modelo y se evalúa en la partición test.
		- Se guarda el "score" y se descarta el modelo.
- Este proceso debe hacerse para cada grupo, de forma que se vaya haciendo una rotación sobre los grupos de test/train y finalizando con la media de los resultados.

### Leave-one-out
- Esta versión de la validación cruzada utiliza un solo dato para el testing de nuestro modelo. De forma que sería un método **n-fold cross-validation** siendo la $k$ escogida $n$, y siendo $n$ a su vez el total de datos en nuestra muestra.
- Se escogen $k$ pliegues con $n-1$ subconjuntos de entrenamiento y $1$ subconjunto de test. Después se ajusta el modelo en cada capa y se estima la media de error de cada subconjunto de validación de cada pliegue.
- Por ejemplo:

![[leave-one-out-cross-validation.png|500]]