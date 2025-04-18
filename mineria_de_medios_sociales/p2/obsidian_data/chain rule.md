- #chainrule | #regladelacadena

### Explicación
- La regla de la cadena es un pequeño truco algebráico que permite simplificar las operaciones de [[derivatives|derivación]].
- Pongamos las siguientes funciones como ejemplo:
	- $y=x²+1$
	- $z=y³-2$
- La función $z$ "llama" a la función $y$, están enlazadas entre ellas. Para conocer el valor de la función $z$ necesitamos resolver el valor de $y$.
	- $z=(x²+1)³-2$
- Entonces tenemos que la derivada de $z$ con respecto a $x$ es:
	- $\frac{dz}{dx}((x²+1)³-2)=6x(x^2+1)²$
- Pero podemos tomar otro camino, si en vez de hacer la derivada conjunta, hacemos la derivada de cada función de manera separada y luego las multiplicamos, obtenemos el mismo resultado.
	- $\frac{d}{dx}y(x)=2x$
	- $\frac{d}{dy}z(y)=3y³$
	- $\frac{d}{dx}z(x)=2x·3y³=6xy²$
- Tenemos pues que:
	- $\frac{dz}{dx}=\frac{dz}{dy}·\frac{dy}{dx}$