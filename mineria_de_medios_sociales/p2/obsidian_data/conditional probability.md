- #conditionalprobability

# Concepto
- La probabilidad condicional es aquella que mide la certeza de que un evento ocurra dado otro u otros eventos ocurriendo. Se condicionan unos eventos dados otros.
	- $P(A\hspace{2mm}GIVEN\hspace{2mm}B)=P(A|B)$
	- $\large{P(A|B)=\frac{P(A\cap B)}{P(B)}}$
	- Ya que ha ocurrido $B$, todos los resultados de $P(A|B)$ son elementos de $B$. Además, si ocurre $A$ entonces $P(A\cap B)$ es cierto también. El objetivo es calcular $P(A\cap B)$ asumiento que $B$ ha ocurrido, por tanto la fórmula divide la unión de eventos entre los eventos que condicionan.
- La dirección de la condición es importante, pues puede dar lugar a confusión. No es lo mismo la probabilidad de tener cáncer si eres bebedor de café, que decir la probabilidad de ser bebedor de café si tienes cáncer.
- Puede interpretarse de manera errónea, que si eres bebedor casual de café, es probable que obtengas cáncer. La primera sentencia si calcula esa probabilidad, pero la segunda, solo está ofreciendo información de si es probable que se beba café teniendo cáncer, lo cual no está haciendo una relación de causa-efecto con el café y el cáncer. Es importante discernir entre ambos casos y tener en cuenta la dirección de la condición.
- La verosimilitud de $A$ dado $B$ $\rightarrow$ [[likelihood function]] $\rightarrow L(A|B)$.
## Regla fundamental
- $P(A|B)\cdot P(B)=P(A\cap B)$
## Teorema de Bayes
- $P(A|B)=\frac{P(B|A)\cdot P(A)}{P(B)}$
- [[bayes theorem]]
## Probabilidades de intersección y unión condicionales
- Si queremos saber la probabilidad de que ocurra un evento $A$ y un evento $B$, podemos usar la [[joint probabilites|probabilidad de intersección]] o podemos usar información sobre $P(A|B)$ y multiplicarlo por $P(B)$. 
- Esto nos diría que dado el evento $B$ ocurra $B$ y $A$. 
- Si hemos establecido la condición de que la probabilidad solo se aplica a $A$ y a $B$, entonces esta manera es más específica
- Si un evento $A$ no tiene impacto en un evento $B$, entonces la probabilidad condicional $P(B|A)=P(B)$. Esto se conoce como **independencia condicional**.
- Si calculamos la probabilidad de que ocurra un evento $A$ o un evento $B$, pero el evento $A$ afecta a la probabilidad de $B$, entonces:
	- $P(A\hspace{2mm}OR\hspace{2mm}B)=P(A)+P(B)-P(A|B)·P(B)$
- En eventos mutuamente exclusivos sirve también, pues la parte que resta sería $0$.
## Variables
- El concepto de probabilidad condicional se puede aplicar a variables:
	- $\sum_{i=1}^{n}P(A=a_{j}|B=b_{j})=1, \forall b_{j}$
- Dado un valor de $B$, debe darse a la fuerza un valor de $A$.