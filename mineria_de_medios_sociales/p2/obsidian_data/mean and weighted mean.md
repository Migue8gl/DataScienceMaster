- #mean | #weightedmead

### Definición
- Es un **estadístico de localización.**
- La media es el valor promedio de un conjunto de valores. La operación es simple, se suman todos los valores y se divide entre el número total de elementos del conjunto.
- Muestra el "centro de gravedad" del conjunto.
	- $\bar{x}=\frac{x_1+x_2+...+x_n}{n}=\sum\frac{x_i}{n}$ (media de la [[sample|muestra]]).
	- $\mu=\frac{x_1+x_2+...+x_n}{N}=\sum\frac{x_i}{N}$ (media de la [[populations|población]]). 
- La media ponderada es exáctamente igual, pero como su propio nombre indica, se le da más valor a ciertos valores que a otros a través de variables de ponderación -> $w_i$.
	- Media ponderada $=\frac{(x_1·w_1)+(x_2·w_2)+...+(x_n·w_n)}{w_1+w_2+...w_n}$.

### Características
- La fórmula es derivable, por lo que es interesante su uso en métodos analíticos más complejos.
- Es sensible a [[outliers]], por lo que algunos casos aislados pueden influir mucho en el valor de la media.