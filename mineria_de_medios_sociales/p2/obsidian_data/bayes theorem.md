- #bayes | #statistics 

# Concepto
- El **Teorema de Bayes** es una regla fundamental en la teoría de probabilidades y la estadística que describe la probabilidad de un evento, basado en el conocimiento previo de condiciones que podrían estar relacionadas con el evento. Se usa principalmente para actualizar las probabilidades de hipótesis a medida que se obtiene nueva evidencia.
## Fórmula General
- La fórmula del Teorema de Bayes es:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
- Donde:
	- $P(A|B)$ es la **probabilidad posterior** de que ocurra el evento A dado que el evento B ha ocurrido.
	- $P(B|A)$ es la **probabilidad de ver B dado que A ha ocurrido** (probabilidad de la evidencia, si la hipótesis es cierta).
	- $P(A)$ es la **probabilidad previa** de que ocurra A (sin tener en cuenta la evidencia B).
	- $P(B)$ es la **probabilidad total** de que ocurra B (la probabilidad de la evidencia, considerando todas las posibles hipótesis).
## Interpretación Intuitiva
- El Teorema de Bayes nos permite actualizar nuestra **creencia inicial (probabilidad previa)** sobre un evento (como tener una enfermedad) a una **probabilidad posterior** basada en nueva evidencia (como el resultado de un test).
### Ejemplo Médico
- Supongamos que estamos realizando un test para detectar una enfermedad rara. Sabemos lo siguiente:
	- La **probabilidad de tener la enfermedad** en la población es baja (por ejemplo, $1$ en $1000$ personas).
	- El test tiene una **alta sensibilidad** (probabilidad de dar positivo si realmente se tiene la enfermedad).
	- El test tiene una **tasa de falsos positivos** (probabilidad de dar positivo aunque no se tenga la enfermedad).
- A partir de estos datos, podemos calcular la **probabilidad de que una persona realmente tenga la enfermedad**, dado que el test ha dado positivo, utilizando el Teorema de Bayes.
## Notación Común en Clasificación
- Cuando aplicamos el Teorema de Bayes a un problema de clasificación (como en detección de anomalías o enfermedades), podemos escribirlo como:
$$P(\text{Clase} | \text{Datos}) = \frac{P(\text{Datos} | \text{Clase}) \cdot P(\text{Clase})}{P(\text{Datos})}$$

- Donde:
	- $P(\text{Clase})$ es la **probabilidad previa** de cada clase (por ejemplo, tener la enfermedad o no).
	- $P(\text{Datos} | \text{Clase})$ es la **probabilidad de observar los datos** bajo cada clase (como la probabilidad de que el test sea positivo si la persona tiene la enfermedad).
	- $P(\text{Datos})$ es la **probabilidad total** de observar los datos en el conjunto de todos los casos posibles.
## Aplicaciones del Teorema de Bayes
1. **Clasificación Naive Bayes**: Un algoritmo de clasificación basado en el Teorema de Bayes que asume que las características (datos) son independientes entre sí, lo que simplifica los cálculos.
2. **Redes Bayesianas**: Modelos probabilísticos que representan dependencias entre variables mediante un grafo acíclico dirigido.
3. **Inferencia estadística**: Actualizar las creencias sobre una hipótesis a medida que se recogen nuevos datos.
4. **Predicción y toma de decisiones**: Usado para realizar predicciones cuando se cuenta con información previa y nueva evidencia.