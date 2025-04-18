- #statistics | #posthoc

# Introducción
- El objetivo del post-hoc es responder a ciertas preguntas despueś de haber concluido otro estudio.
- Ejemplo: Se realiza un test $A$ para ver si hay al menos un grupo que presente diferencias singnificativas entre los $m$ que hay. Ahora, ¿qué grupo es el que lo hace? Ahí, una vez probado $A$ es cuando se hace $B$, el **post-hoc**.
- Los post-hoc no son infalibles, pues se podrían dar casos como el siguiente:

![[post hoc-example.png]]

- En el primer caso obtenemos resultados que hay diferencias entre algunos grupos. No se puede determinar si el grupo $1$ y $2$ son distintos, pero sí que cualquiera de ellos es distinto a los otros. (**Tiene sentido, aunque incompleto**)
- En el segundo caso no obtenemos ninguna evidencia, pues no se puede concluir si hay uno distinto del resto por transitividad. El $1$ es distinto del $3$, pero como no se puede saber si hay diferencia entre $1$ y $2$ y el $2$ tampoco sabemos si tiene diferencia con el resto, por transitividad, no se puede concluir nada (**Hay un error de tipo II**).
- En el tercer caso ningún test rechaza, hay de nuevo error tipo II. Tenemos evidencias gracias a $A$ de que al menos uno es distinto, pero en el seguno análisis no ha podido demostrar.

# Realización
- Si se tienen $k$ grupos se realizan un total de $k(k-1)/2$ tests.
	- $H_{0_{ij}}$ -> La métrica del grupo $i$ es la misma que la del grupo $j$.
	- $H_{1_{ij}}$ -> Son distintas.
- Se pueden hacer test específicos (con penalización ya incluida en ese test) o test genéricos (penalización introducida manualmente).