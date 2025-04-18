- #associationrules | #fuzzy

# Concepto
- Se usan para representar conceptos. La pertenencia a un conjunto o clasificación se trata con un grado de certeza.
- Las transiciones entre intervalos son graduales y no bruscas.

# Medidas
- Las clásicas medidas de soporte, confianza, etc, cambian su definición cuando se trata de reglas difusas.
## Soporte
- De un *itemset* $X$:
	- $Soporte(X)=\sum_{i}^{N}\frac{\mu_{X}(i)}{N}$
- De una regla $X\rightarrow Y$:
	- $Soporte(X\rightarrow Y)=\sum_{i}^{N}\frac{\mu_{XY}(i)}{N}$
- Donde $\mu_{X}(i)$ es el grado con el que el *itemset* $X$ cubre el ejemplo $i$.
## Confianza de la regla $X\rightarrow Y$
$$\frac{X\rightarrow Y}{Soporte(X)}$$
![[fuzzy association rules-example.png]]

# Extracción
- Hay varias formas:
	- **Realizar las particiones difusas a priori** y extraer a partir de ellas las reglas que cumplan los umbrales de mínimo soporte y mínima confianza. Las particiones pueden ser proporcionadas por el experto o aprendidas mediante un método automático.
	- **Aprender las reglas y particiones difusas las mismo tiempo**.