- #statistics | #requirements

# Tests paramétricos
## Requisito de independencia.
- Necesidad de que las variables estudiadas no dependan o no estén relacionadas entre sí.
- Esto es que la ocurrencia una variable no afecta a la otra u otras variables.
- Las filas de un conjunto deben ser independientes entre sí.
## Requisito de normalidad
- Los datos de cada grupo medido deben seguir una [[normal distribution|distribución normal]].
- En caso contrario se usan test no paramétricos. Se pueden usar gráficas de [[qqplot]].
- Este requisito es sobre los [[residuals|residuos]].
## Requisito de homocedasticidad
- La varianza de los grupos han de ser similares. Se puede comprobar usando gráficos de [[boxplot]].
- Se puede hacer también el test de **Levene** (test de varianzas).

# Tests no paramétricos
- Menos **potentes**. 
- Más robustos frente a [[outliers]], ya que usa la mediana (los que la usan) y no la media.
- Requieren **INDEPENDENCIA**, pero no requieren **NORMALIDAD**.
- Los test no paramétricos son:
	- Tests sobre las [[median|medianas]] si las distribuciones de los distintos grupos son similares.
	- Tests de [[stochastic dominance|dominación estocástica]].
- Si un grupo $A$ tiene una [[median|mediana]] mayor que $B$, entonces $A$ domina estocásticamente sobre $B$, pero no tiene por qué suceder al revés. Por ello, los tests sobre medianas son más restrictivos. Es decir, si la mediana es mayor, entonces domina, pero si domina, no tiene por qué cumplirse que la mediana sea mayor.

# Muestras dependientes
- Deben cumplir con la normalidad, la independencia entre observaciones (filas) y ...
## Esfericidad
- Equivalente a la homocedasticidad y además las covarianzas entre todos los pares de grupos también han de ser similares. Se comprueba con el test de **Mauchy**. Si no, se aplica una correción sobre los grados de libertad -> Correción **Greenhouse-Geisser**.