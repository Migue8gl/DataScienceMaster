- #hypothesis | #statistics 

# Errores
- El error de tipo I ocurre cuando se rechaza incorrectamente la hipótesis nula, cuando es realidad es cierta.
- El error de tipo II ocurre cuando no se rechaza una hipótesis nula, es decir, cuando la hipótesis alternativa es cierta. Cuando no se tiene **potencia** para detectar un efecto.
![[hypothesis errors-table-example.png]]
- ¿Cuál es la probabilidad de que se rechace erróneamente (**falso positivo**)?
	- **Error tipo I** -> $P(\text{rechazar }H_0|H_{0}\text{ es cierta})=\alpha$
	- Positivo (el test ha decidido rechazar).
- ¿Cuál es la probabilidad de no se rechace cuando se debería haber hecho (**falso negativo**)?
	- **Error tipo II** -> $P(\text{No rechazar }H_0|H_{0}\text{ es false})=P(\text{rechazar }H_1|H_{1}\text{ es cierta})=\beta$
	- Positivo (el test no rechaza).

# Potencia
- $Potencia=P(\text{rechazar }H_0|H_{1}\text{ es cierta}) = 1-\beta$
- Un test **conservador** no tendrá mucha potencia.
- Un test que no rechaza todas las veces que debería haber rechazado no tiene mucha potencia. Un test con baja potencia es muy sensible a los errores tipo II.

# Detalles
- El error tipo $\beta$ depende de:
	- La hipótesis nula que se plantea.
	- La calidad del estadístico.
	- El tamaño muestral (mayor es mejor).
	- La varianza de los datos (cuando menor mejor).