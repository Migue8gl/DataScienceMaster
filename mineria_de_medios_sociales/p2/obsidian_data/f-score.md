- #fscore

# Concepto
- El **F-score** (también conocido como **F1-score**) es una métrica de evaluación que combina la precisión y la exhaustividad (*recall*) de un modelo de clasificación. Se usa especialmente en problemas donde se busca un equilibrio entre ambos y es útil cuando tenemos clases desbalanceadas (es decir, cuando hay muchos más ejemplos de una clase que de otra). Su fórmula es:
$$F_1 = 2 \times \frac{\text{Precisión} \times \text{Exhaustividad}}{\text{Precisión} + \text{Exhaustividad}}$$
# Interpretación
- El F1-score tiene un valor entre $0$ y $1$:
- Un **F1-score de 1** indica que el modelo tiene tanto precisión como exhaustividad perfectas.
- Un **F1-score cercano a 0** indica que el modelo no tiene un buen desempeño ni en precisión ni en exhaustividad.

# Ventajas del F1-score
- Es útil cuando las clases están desbalanceadas, ya que considera tanto los falsos positivos como los falsos negativos.
- Es más representativo que solo la precisión o solo la exhaustividad en problemas donde ambos son importantes.

# Otros
- En algunos casos, si deseas ajustar la importancia relativa entre precisión y exhaustividad, puedes usar el **F-beta score**, que generaliza el F1-score para dar más peso a uno u otro:
$$F_\beta = (1 + \beta^2) \times \frac{\text{Precisión} \times \text{Exhaustividad}}{(\beta^2 \times \text{Precisión}) + \text{Exhaustividad}}$$
- Donde:
	- Si $\beta > 1$, se da más peso a la exhaustividad.
	- Si $\beta<1$ se da más peso a la precisión.