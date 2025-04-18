- #unionprobabilities

- Este tipo de probabilidad mide la probabilidad de que ocurra un evento $A$ o un evento $B$, lo que se conocería también como una operación $OR$.

### Eventos mutuamente exclusivos
- Son aquellos que no pueden ocurrir de manera simultánea, o ocurre uno, o ocurre el otro.
	- $P(A\hspace{2mm}OR\hspace{2mm}B)=P(A)+P(B)$

### Eventos mutuamente no exclusivos
- Son aquellos que sí pueden ocurrir a la vez, puede ocurrir el evento $A$, el $B$ y los dos a la vez, a diferencia de los mutuamente exclusivos.
- Si hacemos la operación anterior tenemos el problema de estar contando varias veces el mismo evento, al sumar $P(A)$ y $P(B)$. Debemos en estos casos quitar la probabilidad de aquellos casos repetidos:
	- $P(A\hspace{2mm}OR\hspace{2mm}B)=P(A)+P(B)-P(A\hspace{2mm}AND\hspace{2mm}B)$

![[disjunction.png]]